import enum
import time
import torch
import torch.distributed as dist
from .compression import TopkLayer, QSendLayerGPU, QRecvLayerGPU
from torch.optim import AdamW, SGD
from transformers import get_scheduler
import torch.nn as nn
from .utils import SendTensor, get_lr, accuracy, RecvTensor


def init_models_server(train_settings, server_settings):
    param_list = []
    group_list = []
    for chunk in range(server_settings["chunks"]):
        group_list.append(dist.new_group(ranks=server_settings["devices"]))
    for model in train_settings["models"]:
        model = model.to(train_settings["device"], non_blocking=True)
        # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[server_settings["device"]])
        param_list.append({"params": model.parameters()})
    if train_settings["tasktype"] == "cv":
        optimizer = SGD(
            param_list, lr=train_settings["lr"], weight_decay=train_settings["wd"]
        )
        warmup_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=int(train_settings["epochs"] / 10),
            num_training_steps=train_settings["epochs"],
        )
    elif train_settings["tasktype"] == "nlp":
        optimizer = AdamW(
            param_list, lr=train_settings["lr"], weight_decay=train_settings["wd"]
        )
        warmup_scheduler = get_scheduler(
            name="polynomial",
            optimizer=optimizer,
            num_warmup_steps=int(
                train_settings["epochs"] / 10 * train_settings["len_trainloader"]
            ),
            num_training_steps=train_settings["epochs"]
            * train_settings["len_trainloader"],
        )
    topk_layer = TopkLayer(train_settings["prune"], server_settings["send_size"])
    quant_layer = QSendLayerGPU(
        train_settings["quant"], server_settings["send_rank"], server_settings["rank"]
    )
    dequant_layer = QRecvLayerGPU(
        train_settings["quant"], server_settings["recv_rank"], server_settings["rank"]
    )
    return (
        topk_layer,
        quant_layer,
        dequant_layer,
        optimizer,
        warmup_scheduler,
        group_list,
    )


def server_trainer(
    train_settings,
    server_settings,
    topk_layer,
    quant_layer,
    dequant_layer,
    optimizer,
    warmup_scheduler,
):
    if train_settings["tasktype"] == "cv":
        for batch_iter in range(train_settings["len_trainloader"]):
            batches = []
            for i, model in enumerate(train_settings["models"]):
                model.train()
                batch = []
                for chunk in range(server_settings["chunks"]):
                    input = (
                        torch.zeros(server_settings["recv_size"])
                        .to(server_settings["device"])
                        .requires_grad_()
                    )
                    input = RecvTensor(input, server_settings, train_settings)
                    # print("server",server_settings['rank'],"recv",server_settings['recv_rank'],input.shape)
                    output = model(input)

                    output = SendTensor(output, server_settings, train_settings)
                    # print("server",server_settings['rank'],"send",server_settings['send_rank'],output.shape)
                    batch.append(output)
                batches.append(batch)
            for back in range(len(train_settings["models"]) - 1, -1, -1):
                for chunk in range(server_settings["chunks"]):

                    batches[back][chunk].backward(
                        torch.empty(tuple(list(batches[back][chunk].shape))).to(
                            server_settings["device"]
                        )
                    )
            optimizer.step()
            optimizer.zero_grad()
    else:
        for batch_iter in range(train_settings["len_trainloader"]):
            batches = []
            attention_mask = (
                torch.zeros(
                    server_settings["recv_size"][0] * server_settings["chunks"],
                    1,
                    1,
                    server_settings["recv_size"][1],
                )
                .type(torch.float32)
                .to(server_settings["rank"])
            )
            dist.recv(attention_mask, 0)
            # print("server rev mask")
            attention_mask = attention_mask.chunk(server_settings["chunks"])
            for i, model in enumerate(train_settings["models"]):
                model.train()
                batch = []
                for chunk in range(server_settings["chunks"]):
                    input = (
                        torch.zeros(server_settings["recv_size"])
                        .to(server_settings["device"])
                        .requires_grad_()
                        # .type(torch.long)
                    )

                    output = RecvTensor(input, server_settings, train_settings)
                    output = model(output, attention_mask[chunk])

                    output = SendTensor(output, server_settings, train_settings)
                    batch.append(output)
                batches.append(batch)
            # print("server forward finish",server_settings["rank"])
            for back in range(len(train_settings["models"]) - 1, -1, -1):
                for chunk in range(server_settings["chunks"]):

                    batches[back][chunk].backward(
                        torch.empty(tuple(list(batches[back][chunk].shape))).to(
                            server_settings["device"]
                        )
                    )
            optimizer.step()
            optimizer.zero_grad()
            warmup_scheduler.step()
            # print("server batch_iter",server_settings["rank"])


def server_validation(
    train_settings, server_settings, topk_layer, quant_layer, dequant_layer
):
    if train_settings["tasktype"] == "cv":
        for model in train_settings["models"]:
            model.eval()
        with torch.no_grad():
            for batch_iter in range(train_settings["len_valloader"]):
                for i, model in enumerate(train_settings["models"]):
                    #   model.eval()
                    for chunk in range(server_settings["chunks"]):
                        input = torch.zeros(server_settings["recv_size"]).to(
                            server_settings["device"]
                        )
                        input = RecvTensor(input, server_settings, train_settings)

                        output = model(input)
                        output = SendTensor(output, server_settings, train_settings)

    else:
        for model in train_settings["models"]:
            model.eval()
        with torch.no_grad():
            for batch_iter in range(train_settings["len_valloader"]):
                # print("server",server_settings["rank"],batch_iter)
                attention_mask = (
                    torch.zeros(
                        server_settings["recv_size"][0] * server_settings["chunks"],
                        1,
                        1,
                        server_settings["recv_size"][1],
                    )
                    .type(torch.float32)
                    .to(server_settings["rank"])
                )
                dist.recv(attention_mask, 0)
                # print("server rev mask")
                attention_mask = attention_mask.chunk(server_settings["chunks"])
                for i, model in enumerate(train_settings["models"]):
                    for chunk in range(server_settings["chunks"]):
                        input = (
                            torch.zeros(server_settings["recv_size"]).to(
                                server_settings["device"]
                            )
                            # .type(torch.long)
                        )

                        input = RecvTensor(input, server_settings, train_settings)
                        # input = input.type(torch.long)
                        # attention_mask = (
                        #     torch.zeros(input.shape[0],1,1,input.shape[-1])
                        #     .type(torch.long)
                        #     .to(server_settings["rank"])
                        # )
                        # dist.recv(attention_mask,0)
                        # attention_mask[input != 0] = 1
                        # attention_mask = torch.reshape(
                        #     attention_mask,
                        #     [
                        #         int(attention_mask.shape[0]),
                        #         1,
                        #         1,
                        #         int(attention_mask.shape[-1]),
                        #     ],
                        # ).to(server_settings["rank"])
                        # attention_mask = (1.0 - attention_mask) * -1e4
                        output = model(input, attention_mask[chunk])

                        output = SendTensor(output, server_settings, train_settings)


def server(train_settings, server_settings):
    torch.cuda.set_device(server_settings["device"])
    # maybe not right
    torch.multiprocessing.set_sharing_strategy("file_system")
    print(
        "server",
        server_settings["backend"],
        server_settings["dist_url"],
        server_settings["world_size"],
        server_settings["rank"],
    )
    dist.init_process_group(
        backend=server_settings["backend"],
        init_method=server_settings["dist_url"],
        world_size=server_settings["world_size"],
        rank=server_settings["rank"],
    )
    print("process begin: ", server_settings["rank"])
    (
        topk_layer,
        quant_layer,
        dequant_layer,
        optimizer,
        warmup_scheduler,
        group_list,
    ) = init_models_server(train_settings, server_settings)
    server_settings["group_list"] = group_list
    for epoch in range(train_settings["epochs"]):
        server_trainer(
            train_settings,
            server_settings,
            topk_layer,
            quant_layer,
            dequant_layer,
            optimizer,
            warmup_scheduler,
        )
        if train_settings["tasktype"] == "cv":
            warmup_scheduler.step()
        server_validation(
            train_settings, server_settings, topk_layer, quant_layer, dequant_layer
        )
