# prune quant 这些不用的时候为0不是None
import time
from tokenize import group
import torch
import torch.distributed as dist
from .compression import TopkLayer, QSendLayerGPU, QRecvLayerGPU
from torch.optim import AdamW, SGD
from transformers import get_scheduler
import torch.nn as nn
from .utils import SendTensor, get_lr, accuracy, RecvTensor


def init_models_client(train_settings, client_settings):
    param_list = []
    group_list = []
    print(client_settings["devices"])
    for chunk in range(client_settings["chunks"]):
        group_list.append(dist.new_group(ranks=client_settings["devices"]))
    for model in train_settings["models"]:
        model = model.to(client_settings["device"], non_blocking=True)
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
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=int(
                train_settings["epochs"] / 10 * len(train_settings["train_loader"])
            ),
            num_training_steps=train_settings["epochs"]
            * len(train_settings["train_loader"]),
        )
    criterion = nn.CrossEntropyLoss().to(client_settings["device"])
    return (optimizer, warmup_scheduler, criterion, group_list)


def client_trainer(
    train_settings, client_settings, optimizer, warmup_scheduler, criterion,
):
    acc1_avg = 0.0
    losses_avg = 0.0
    train_acc1_avg = 0.0
    train_losses_avg = 0.0
    time_per_batch = 0.0
    batch_time = 0.0
    bandwidth = torch.tensor([0.0])
    bandwidth_avg = 0.0
    if train_settings["tasktype"] == "cv":
        start = time.time()
        for batch_iter, (images, targets) in enumerate(train_settings["train_loader"]):

            images = images.to(client_settings["device"], non_blocking=True)
            targets = targets.to(client_settings["device"], non_blocking=True)
            images = images.chunk(client_settings["chunks"])
            targets = targets.chunk(client_settings["chunks"])
            batches = []
            acc1 = 0.0
            losses = 0.0
            for i, model in enumerate(train_settings["models"]):
                batch = []
                model.train()
                for chunk in range(client_settings["chunks"]):
                    if i == 0:
                        output = model(images[chunk])

                        output = SendTensor(
                            output, client_settings, train_settings, chunk, True
                        )

                        # print("client",client_settings['rank'],"send",output.shape)
                    else:
                        input = (
                            torch.zeros(client_settings["recv_size"])
                            .to(client_settings["device"])
                            .requires_grad_()
                        )

                        input = RecvTensor(
                            input,
                            client_settings,
                            train_settings,
                            chunk,
                            True,
                            bandwidth,
                        )
                        # print("client",client_settings['rank'],"recv",input.shape)
                        output = model(input)
                        acc, _ = accuracy(output, targets[chunk], topk=(1, 2))
                        output = criterion(output, targets[chunk])
                        losses += output.item()
                        acc1 = acc1 + acc.item()
                        bandwidth_avg += bandwidth.item()
                    batch.append(output)
                batches.append(batch)
            # bandwidth /= client_settings["chunks"]
            acc1 = acc1 / client_settings["chunks"]
            losses = losses / client_settings["chunks"]
            acc1_avg, losses_avg = acc1 + acc1_avg, losses_avg + losses

            if batch_iter % client_settings["showperiod"] == 0:
                print(
                    "training_loss:",
                    losses,
                    "training_acc",
                    acc,
                    "bandwidth",
                    bandwidth.item(),
                )

            for back in range(len(train_settings["models"]) - 1, -1, -1):
                if back == len(train_settings["models"]) - 1:
                    for chunk in range(client_settings["chunks"]):
                        batches[back][chunk].backward()
                else:
                    for chunk in range(client_settings["chunks"]):
                        batches[back][chunk].backward(
                            torch.empty(tuple(list(batches[back][chunk].shape))).to(
                                client_settings["device"]
                            )
                        )
            optimizer.step()
            optimizer.zero_grad()
            batch_time = time.time() - start
            time_per_batch += batch_time
            start = time.time()
        time_per_batch = time_per_batch / len(train_settings["train_loader"])
        train_acc1_avg, train_losses_avg = (
            acc1_avg / len(train_settings["train_loader"]),
            losses_avg / len(train_settings["train_loader"]),
        )
        bandwidth_avg = (
            bandwidth_avg
            / len(train_settings["train_loader"])
            / client_settings["chunks"]
        )

    else:
        start = time.time()
        bandwidth = torch.tensor([0.0])
        bandwidth_avg = 0.0
        for batch_iter, batch in enumerate(train_settings["train_loader"]):
            # print("client batch_iter")
            batch = {
                k: v.to(client_settings["rank"], non_blocking=True)
                for k, v in batch.items()
            }
            acc1 = 0.0
            losses = 0.0
            batch["attention_mask"] = (
                torch.reshape(
                    batch["attention_mask"],
                    [
                        int(batch["attention_mask"].shape[0]),
                        1,
                        1,
                        int(batch["attention_mask"].shape[-1]),
                    ],
                )
                .to(client_settings["rank"])
                .type(torch.float32)
            )
            batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e4
            for rank in client_settings["devices"]:
                if rank != client_settings["rank"]:
                    dist.isend(batch["attention_mask"], rank)
            batch["attention_mask"] = batch["attention_mask"].chunk(
                client_settings["chunks"]
            )
            batch["input_ids"] = batch["input_ids"].chunk(client_settings["chunks"])
            batch["labels"] = batch["labels"].chunk(client_settings["chunks"])
            batches = []

            for i, model in enumerate(train_settings["models"]):
                batch_save = []
                model.train()
                for chunk in range(client_settings["chunks"]):
                    if i == 0:
                        output = model(
                            batch["input_ids"][chunk], batch["attention_mask"][chunk]
                        )

                        output = SendTensor(
                            output, client_settings, train_settings, chunk, True
                        )

                    else:
                        input = (
                            torch.zeros(client_settings["recv_size"])
                            .to(client_settings["device"])
                            .requires_grad_()
                            # .type(torch.long)
                        )

                        input = RecvTensor(
                            input,
                            client_settings,
                            train_settings,
                            chunk,
                            True,
                            bandwidth,
                        )

                        # input = input.type(torch.long)
                        output = model(input, batch["attention_mask"][chunk])
                        acc, _ = accuracy(output, batch["labels"][chunk], topk=(1, 2))
                        output = criterion(output, batch["labels"][chunk])
                        losses += output.item()
                        acc1 = acc1 + acc.item()
                        bandwidth_avg += bandwidth.item()
                    batch_save.append(output)
                batches.append(batch_save)
                acc1 = acc1 / client_settings["chunks"]
                losses = losses / client_settings["chunks"]
                # bandwidth /= client_settings["chunks"]
            if batch_iter % client_settings["showperiod"] == 0:
                print(
                    "training_loss:",
                    losses,
                    "training_acc:",
                    acc1,
                    "bandwidth:",
                    bandwidth,
                )
            acc1_avg, losses_avg = acc1 + acc1_avg, losses_avg + losses
            # bandwidth_avg +=bandwidth
            # print("client forward over")
            for back in range(len(train_settings["models"]) - 1, -1, -1):
                if back == len(train_settings["models"]) - 1:
                    for chunk in range(client_settings["chunks"]):

                        batches[back][chunk].backward()
                else:
                    for chunk in range(client_settings["chunks"]):
                        batches[back][chunk].backward(
                            torch.empty(tuple(list(batches[back][chunk].shape))).to(
                                client_settings["device"]
                            )
                            # .type(torch.long)
                        )
            # print("client backword over")
            optimizer.step()
            optimizer.zero_grad()
            warmup_scheduler.step()
            batch_time = time.time() - start
            # warmup_scheduler.step()
            time_per_batch += batch_time
            start = time.time()
        time_per_batch = time_per_batch / len(train_settings["train_loader"])

        train_acc1_avg, train_losses_avg = (
            acc1_avg / len(train_settings["train_loader"]),
            losses_avg / len(train_settings["train_loader"]),
        )
        bandwidth_avg = (
            bandwidth_avg
            / len(train_settings["train_loader"])
            / client_settings["chunks"]
        )
    return (
        time_per_batch,
        train_acc1_avg,
        train_acc1_avg,
        train_losses_avg,
        bandwidth_avg,
    )


def client_validation(train_settings, client_settings, criterion):
    if train_settings["tasktype"] == "cv":
        for model in train_settings["models"]:
            model.eval()
        val_acc_avg = 0.0
        val_loss_avg = 0.0
        with torch.no_grad():
            for batch_iter, (images, targets) in enumerate(train_settings["valloader"]):
                images = images.to(client_settings["device"], non_blocking=True)
                targets = targets.to(client_settings["device"], non_blocking=True)
                images = images.chunk(client_settings["chunks"])
                targets = targets.chunk(client_settings["chunks"])
                acc1 = 0.0
                losses = 0.0
                for i, model in enumerate(train_settings["models"]):
                    for chunk in range(client_settings["chunks"]):
                        if i == 0:
                            output = model(images[chunk])

                            output = SendTensor(
                                output, client_settings, train_settings, chunk, True
                            )
                        else:

                            input = (
                                torch.empty(client_settings["recv_size"])
                                .to(client_settings["device"])
                                .requires_grad_()
                            )

                            input = RecvTensor(
                                input, client_settings, train_settings, chunk, True
                            )
                            output = model(input)
                            acc, _ = accuracy(output, targets[chunk], topk=(1, 2))
                            output = criterion(output, targets[chunk])
                            losses = losses + output.item()
                            acc1 = acc1 + acc.item()
                acc1 = acc1 / client_settings["chunks"]
                losses = losses / client_settings["chunks"]
                if batch_iter % client_settings["showperiod"] == 0:
                    print("val_loss:", losses, "val_acc:", acc1)
                val_acc_avg, val_loss_avg = acc1 + val_acc_avg, losses + val_loss_avg
            val_acc_avg, val_loss_avg = (
                val_acc_avg / len(train_settings["valloader"]),
                val_loss_avg / len(train_settings["valloader"]),
            )
    else:
        for model in train_settings["models"]:
            model.eval()
        val_acc_avg = 0.0
        val_loss_avg = 0.0
        with torch.no_grad():
            for batch_iter, batch in enumerate(train_settings["valloader"]):
                # print("client",client_settings["rank"],batch_iter)
                batch = {k: v.to(client_settings["rank"]) for k, v in batch.items()}
                acc1 = 0.0
                losses = 0.0
                batch["attention_mask"] = torch.reshape(
                    batch["attention_mask"],
                    [
                        int(batch["attention_mask"].shape[0]),
                        1,
                        1,
                        int(batch["attention_mask"].shape[-1]),
                    ],
                ).to(client_settings["rank"])
                batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e4
                # dist.isend(batch["attention_mask"],1)
                # # dist.isend(batch["attention_mask"],2)
                # # dist.isend(batch["attention_mask"],3)
                for rank in client_settings["devices"]:
                    if rank != client_settings["rank"]:
                        dist.isend(batch["attention_mask"], rank)
                batch["attention_mask"] = batch["attention_mask"].chunk(
                    client_settings["chunks"]
                )
                batch["input_ids"] = batch["input_ids"].chunk(client_settings["chunks"])
                batch["labels"] = batch["labels"].chunk(client_settings["chunks"])
                # batches = []
                for i, model in enumerate(train_settings["models"]):
                    # batch = []
                    # model.train()
                    for chunk in range(client_settings["chunks"]):
                        if i == 0:
                            output = model(
                                batch["input_ids"][chunk],
                                batch["attention_mask"][chunk],
                            )

                            output = SendTensor(
                                output, client_settings, train_settings, chunk, True
                            )
                        else:
                            input = (
                                torch.zeros(client_settings["recv_size"])
                                .to(client_settings["device"])
                                .requires_grad_()
                                # .type(torch.long)
                            )

                            input = RecvTensor(
                                input, client_settings, train_settings, chunk, True
                            )
                            # input = input.type(torch.long)
                            output = model(input, batch["attention_mask"][chunk])
                            acc, _ = accuracy(
                                output, batch["labels"][chunk], topk=(1, 2)
                            )
                            output = criterion(output, batch["labels"][chunk])
                            losses += output.item()
                            acc1 = acc1 + acc.item()
                acc1 = acc1 / client_settings["chunks"]
                losses = losses / client_settings["chunks"]
                if batch_iter % client_settings["showperiod"] == 0:
                    print("val_loss:", losses, "val_acc:", acc1)

                val_acc_avg, val_loss_avg = acc1 + val_acc_avg, losses + val_loss_avg
            val_acc_avg, val_loss_avg = (
                val_acc_avg / len(train_settings["valloader"]),
                val_loss_avg / len(train_settings["valloader"]),
            )
    return val_acc_avg, val_acc_avg, val_loss_avg


def client(train_settings, client_settings):
    s = torch.cuda.Stream(device=client_settings["device"])
    with torch.cuda.stream(s):
        torch.cuda.set_device(client_settings["device"])

        # may be not right
        torch.multiprocessing.set_sharing_strategy("file_system")
        print(
            "client",
            client_settings["backend"],
            client_settings["dist_url"],
            client_settings["world_size"],
            client_settings["rank"],
        )
        dist.init_process_group(
            backend=client_settings["backend"],
            init_method=client_settings["dist_url"],
            world_size=client_settings["world_size"],
            rank=client_settings["rank"],
        )
        print("process begin: ", client_settings["rank"])
        (optimizer, warmup_scheduler, criterion, group_list) = init_models_client(
            train_settings, client_settings
        )
        client_settings["group_list"] = group_list
        print("client", group_list)
        for epoch in range(train_settings["epochs"]):
            (
                train_time,
                train_acc,
                train_metric,
                train_loss,
                bandwidth_avg,
            ) = client_trainer(
                train_settings, client_settings, optimizer, warmup_scheduler, criterion,
            )
            if train_settings["tasktype"] == "cv":
                warmup_scheduler.step()
            val_acc, val_metric, val_loss = client_validation(
                train_settings, client_settings, criterion,
            )
            print(
                "epoch",
                epoch,
                "train_acc",
                train_acc,
                "train_loss",
                train_loss,
                "val_acc",
                val_acc,
                "val_loss",
                val_loss,
                "lr",
                get_lr(optimizer),
            )
            file_save = open(client_settings["savepath"], mode="a")
            file_save.write(
                "\n"
                + "step:"
                + str(epoch)
                + "  loss_train:"
                + str(train_loss)
                + "  acc1_train:"
                + str(train_acc)
                + "  loss_val:"
                + str(val_loss)
                + "  acc1_val:"
                + str(val_acc)
                + "  time_per_batch:"
                + str(train_time)
                + "  lr:"
                + str(get_lr(optimizer))
                + "  bandwidth:"
                + str(bandwidth_avg)
            )
            file_save.close()
