import enum
import time
import torch
import torch.distributed as dist
from .compression import (
    TopkLayer,
    QSendLayerGPU,
    QRecvLayerGPU,
    PowerSVDServerSendLayer,
    PowerSVDServerRecvLayer,
)
from torch.optim import AdamW, SGD
from transformers import get_scheduler
import torch.nn as nn
from .utils import SendTensor, RecvTensor
import os


def init_models_server(train_settings, server_settings):
    param_list = []
    group_list = []
    # print("server",server_settings['devices'])
    for chunk in range(server_settings["chunks"]):
        group_list.append(dist.new_group(ranks=server_settings["ranks"]))
    for model in train_settings["models"]:
        model = model.to(train_settings["device"], non_blocking=True)
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
            param_list,
            lr=train_settings["lr"],
            weight_decay=train_settings["wd"],
            eps=1e-4,
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
    if train_settings["poweriter2"] != 0:
        train_settings["poweriter2_layer"] = PowerSVDServerSendLayer(
            train_settings["poweriter2"],
            server_settings["send_size"],
            3,
            server_settings["send_rank"],
            server_settings["device"],
        ).to(server_settings["device"])
    if train_settings["poweriter1"] != 0:
        train_settings["poweriter1_layer"] = PowerSVDServerRecvLayer(
            train_settings["poweriter1"],
            server_settings["recv_size"],
            3,
            server_settings["recv_rank"],
            server_settings["device"],
        ).to(server_settings["device"])
    return optimizer, warmup_scheduler, group_list


def server_trainer(
    train_settings,
    server_settings,
    optimizer,
    warmup_scheduler,
):
    if train_settings["tasktype"] == "cv":
        timerecv_avg = 0.0
        timecount = torch.tensor([0.0])
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

                    input = RecvTensor(
                        input, server_settings, train_settings, chunk, False, timecount
                    )
                    # print("server, recv", chunk)
                    # print("server",server_settings['rank'],"recv",server_settings['recv_rank'],input.shape)
                    output = model(input)
                    # output = output.view([output.shape[0],output.shape[1],49])
                    output = SendTensor(output, server_settings, train_settings, chunk)
                    # print("server, send", chunk)
                    # print("server",server_settings['rank'],"send",server_settings['send_rank'],output.shape)
                    timerecv_avg += timecount.item()
                    # output = output.cpu()
                    batch.append(output)
                batches.append(batch)
                # timecount /= chunk
            # torch.cuda.synchronize()
            # forward_time = time.time() - start
            for back in range(len(train_settings["models"]) - 1, -1, -1):
                for chunk in range(server_settings["chunks"]):
                    # print("server backward pre",chunk)
                    # batches[back][chunk] = batches[back][chunk].to(server_settings["device"])
                    batches[back][chunk].backward(
                        torch.empty(tuple(list(batches[back][chunk].shape))).to(
                            server_settings["device"]
                        )
                    )
                    # print("server backward",chunk)
            # torch.cuda.synchronize()
            # end = time.time() - start
            # if batch_iter % server_settings["showperiod"] == 0:
            #     print("server_time", end, "forward_time", forward_time)

            optimizer.step()
            optimizer.zero_grad()
        timerecv_avg = (
            timerecv_avg / train_settings["len_trainloader"] / server_settings["chunks"]
        )
        # print("server avg bandwidth:", timerecv_avg)
    else:
        timerecv_avg = 0.0
        timecount = torch.tensor([0.0])
        for batch_iter in range(train_settings["len_trainloader"]):
            batches = []
            attention_mask = torch.zeros(
                server_settings["recv_size"][0] * server_settings["chunks"],
                1,
                1,
                server_settings["recv_size"][1],
            ).to(server_settings["rank"])

            attention_mask = attention_mask.type(torch.float32)
            dist.recv(attention_mask, 0)
            # print("server rev mask",server_settings["rank"])
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
                    # print("server recv1",input.shape)
                    output = RecvTensor(
                        input, server_settings, train_settings, chunk, False, timecount
                    )
                    # print("server recv3",output.shape)
                    output = model(output, attention_mask[chunk])

                    output = SendTensor(output, server_settings, train_settings, chunk)
                    timerecv_avg = timerecv_avg + timecount.item()
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
        timerecv_avg = (
            timerecv_avg / train_settings["len_trainloader"] / server_settings["chunks"]
        )
        print("server avg bandwidth:", timerecv_avg)


def server_validation(train_settings, server_settings):
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
                        input = RecvTensor(
                            input, server_settings, train_settings, chunk
                        )

                        output = model(input)
                        output = SendTensor(
                            output, server_settings, train_settings, chunk
                        )

    else:
        for model in train_settings["models"]:
            model.eval()
        with torch.no_grad():
            for batch_iter in range(train_settings["len_valloader"]):
                # print("server",server_settings["rank"],batch_iter)
                attention_mask = torch.zeros(
                    server_settings["recv_size"][0] * server_settings["chunks"],
                    1,
                    1,
                    server_settings["recv_size"][1],
                ).to(server_settings["rank"])
                if train_settings["fp16"] != 0:
                    attention_mask = attention_mask.type(torch.float16)
                else:
                    attention_mask = attention_mask.type(torch.float32)
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

                        input = RecvTensor(
                            input, server_settings, train_settings, chunk
                        )
                        output = model(input, attention_mask[chunk])

                        output = SendTensor(
                            output, server_settings, train_settings, chunk
                        )


def server(train_settings, server_settings):
    s = torch.cuda.Stream(device=server_settings["device"])
    os.environ["NCCL_SOCKET_IFNAME"] = server_settings["ifconfig"]
    os.environ["NCCL_IB_DISABLE"] = "1"
    with torch.cuda.stream(s):
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
        (optimizer, warmup_scheduler, group_list) = init_models_server(
            train_settings, server_settings
        )
        server_settings["group_list"] = group_list
        # print("server",group_list)
        for epoch in range(train_settings["epochs"]):
            server_trainer(
                train_settings,
                server_settings,
                optimizer,
                warmup_scheduler,
            )
            if train_settings["tasktype"] == "cv":
                warmup_scheduler.step()
            server_validation(train_settings, server_settings)
