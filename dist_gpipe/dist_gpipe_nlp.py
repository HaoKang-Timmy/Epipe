import torch
import time
import torch.nn as nn
from typing import List
from transformers import get_scheduler
from .compression.compression_layer import (
    QuantizationLayer,
    RemoteDeQuantizationLayer,
    RemoteQuantizationLayer,
    TopkLayer,
)
from .distributedlayers import DeferredBatchNorm
import torch.multiprocessing as mp

# import torch.multiprocessing as mp
# import threading
from .distributedlayers import (
    ForwardReceive_BackwardSend,
    ForwardSend_BackwardReceive,
    ForwardSendLayers,
    ForwardReceiveLayers,
)
import torch.distributed as dist
import pytorch_warmup as warmup
from utils import accuracy
import torchvision
import torchvision.transforms as transforms


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def worker_header(
    rank,
    device,
    models,
    chunks,
    criterion,
    backend,
    dist_url,
    world_size,
    recev_size,
    epochs,
    optimizer,
    scheduler,
    savepth,
    warm_up,
    train_loader,
    val_loader,
    settings,
):
    # header and holds the last layer
    torch.multiprocessing.set_sharing_strategy("file_system")
    print("process begin: ", rank)
    dist.init_process_group(
        backend=backend, init_method=dist_url, world_size=world_size, rank=rank
    )
    # nothing = get_scheduler("linear")
    # transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # trainset = torchvision.datasets.CIFAR10(
    # root='./data', train=True, download=True, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(
    # trainset, batch_size=512, shuffle=True, num_workers=12,drop_last = True)

    # testset = torchvision.datasets.CIFAR10(
    # root='./data', train=False, download=True, transform=transform_test)
    # val_loader = torch.utils.data.DataLoader(
    # testset, batch_size=512, shuffle=False, num_workers=12,drop_last = True)

    for model in models:
        model = model.to(device, non_blocking=True)
    topk_layer = TopkLayer(settings["prun"])
    quant_layer = RemoteQuantizationLayer(settings["quant"], rank + 1, rank + 1)
    dequant_layer = RemoteDeQuantizationLayer(settings["quant"], 3, 3)
    for epoch in range(epochs):
        acc1_avg = 0.0
        losses_avg = 0.0
        train_acc1_avg = 0.0
        train_losses_avg = 0.0
        time_per_batch = 0.0
        batch_time = 0.0
        for batch_iter, batch in enumerate(train_loader):
            start = time.time()
            # images = images.to(device, non_blocking=True)
            # targets = targets.to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            batch["attention_mask"] = torch.reshape(
                batch["attention_mask"],
                [
                    int(batch["attention_mask"].shape[0]),
                    1,
                    1,
                    int(batch["attention_mask"].shape[-1]),
                ],
            ).to(0)
            attention_mask = batch["attention_mask"]
            # attention_mask = attention_mask.chunk(chunks)
            # print("begin broadcast")
            # dist.broadcast(attention_mask,src = rank,async_op=True)#TODO recv
            dist.isend(attention_mask, 1)
            dist.isend(attention_mask, 2)
            dist.isend(attention_mask, 3)
            # print("end")

            input_ids = input_ids.chunk(chunks)
            # print("input_ids",len(input_ids))
            labels = labels.chunk(chunks)

            batches = []
            acc1 = 0.0
            losses = 0.0
            # print("prepare over")
            # forward
            quant_layer.train()
            dequant_layer.train()
            for i, model in enumerate(models):
                batch = []
                # model =model.to(device)
                model.train()
                for j in range(chunks):
                    if i == 0:

                        output = model(input_ids[j])
                        if settings["prun"] != 0:
                            output = topk_layer(output)
                        if settings["quant"] == 0:
                            output = ForwardSend_BackwardReceive.apply(
                                output, rank + 1, rank + 1, rank
                            )
                        else:
                            output = quant_layer(output)
                        # output = ForwardSend_BackwardReceive.apply(output,tuple(list(output.shape)),rank+1,rank+1,rank)
                        # print("forward","rank",rank,"part",i,"finish")
                    elif i == len(models) - 1:
                        # print("last part begin")
                        input = torch.empty(recev_size[0]).to(device).requires_grad_()
                        # input = ForwardReceive_BackwardSend.apply(input,3,3,rank)
                        # print("forward","rank",rank,"part",i,"finish")
                        if settings["quant"] == 0:
                            input = ForwardReceive_BackwardSend.apply(
                                input, 3, 3, 0
                            )  # TODO limited
                        else:
                            input = dequant_layer(input)
                        output = model(input)
                        # print(output)
                        # print("output target shape",output.shape,targets[j].shape)
                        acc, _ = accuracy(output, labels[j], topk=(1, 2))
                        output = criterion(output, labels[j])
                        losses = losses + output.item()
                        acc1 = acc1 + acc.item()
                        # print("rank",rank,"part",i,"input_size",input[0].shape)
                    batch.append(output)
                    # print("rank:",rank,"part:",i,"chunk:",j,"output_size:",batch[0].shape)
                batches.append(batch)
            # forward over
            acc1 = acc1 / chunks
            losses = losses / chunks
            acc1_avg, losses_avg = acc1 + acc1_avg, losses_avg + losses
            if batch_iter % 16 == 0:
                print("tarining_loss:", losses, "training_acc", acc)

            # optimizer.zero_grad()
            # print("forward over")
            # while(1):
            #     pass
            # backward
            for k in range(len(models) - 1, -1, -1):
                if k == len(models) - 1:
                    for l in range(chunks):
                        batches[k][l].backward()
                        # print("rank:",rank,"part:",k,"chunk:",l,"backward")
                else:
                    for l in range(chunks):
                        batches[k][l].backward(
                            torch.empty(tuple(list(batches[k][l].shape))).to(device)
                        )
                        # print("rank:",rank,"part:",k,"chunk:",l,"backward")
            optimizer.step()
            # warm_up.step()
            optimizer.zero_grad()
            batch_time = time.time() - start
            time_per_batch = time_per_batch + batch_time
            # backward over
        time_per_batch = time_per_batch / len(train_loader)
        if settings["warmup"] == 0:
            scheduler.step()
        else:
            scheduler.step(scheduler.last_epoch + 1)
            warm_up.dampen()
        # valdation
        # print("train-finish")
        train_acc1_avg, train_losses_avg = (
            acc1_avg / len(train_loader),
            losses_avg / len(train_loader),
        )
        for model in models:
            model.eval()
        quant_layer.eval()
        dequant_layer.eval()
        val_acc_avg = 0.0
        val_loss_avg = 0.0
        with torch.no_grad():
            # print("validation begin")
            for batch_iter, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                batch["attention_mask"] = torch.reshape(
                    batch["attention_mask"],
                    [
                        int(batch["attention_mask"].shape[0]),
                        1,
                        1,
                        int(batch["attention_mask"].shape[-1]),
                    ],
                ).to(0)
                attention_mask = batch["attention_mask"]

                # dist.broadcast(attention_mask,src = rank,async_op=True)#TODO recv
                dist.isend(attention_mask, 1)
                dist.isend(attention_mask, 2)
                dist.isend(attention_mask, 3)
                attention_mask = attention_mask.chunk(chunks)
                input_ids = input_ids.chunk(chunks)
                labels = labels.chunk(chunks)
                acc1 = 0.0
                losses = 0.0
                for i, model in enumerate(models):
                    # model.eval()

                    for j in range(chunks):
                        if i == 0:

                            output = model(input_ids[j])
                            if settings["quant"] == 0:
                                output = ForwardSend_BackwardReceive.apply(
                                    output, rank + 1, rank + 1, rank
                                )
                            else:
                                output = quant_layer(output)
                            # print("val","forward","epoch:",epochs,"rank:",rank,"part:",i,"chunk:",j)
                        elif i == len(models) - 1:
                            input = (
                                torch.empty(recev_size[0]).to(device).requires_grad_()
                            )
                            if settings["quant"] == 0:
                                input = ForwardReceive_BackwardSend.apply(
                                    input, 3, 3, 0
                                )  # TODO limited
                            else:
                                input = dequant_layer(input)
                            output = model(input)
                            # print("val","forward","epoch:",epochs,"rank:",rank,"part:",i,"chunk:",j)
                            acc, _ = accuracy(output, labels[j], topk=(1, 2))
                            output = criterion(output, labels[j])
                            losses = losses + output.item()
                            acc1 = acc1 + acc.item()
                acc1 = acc1 / chunks
                losses = losses / chunks
                if batch_iter % 16 == 0:
                    print("val_loss:", losses, "val_acc:", acc1)
                val_acc_avg, val_loss_avg = acc1 + val_acc_avg, losses + val_loss_avg

            val_acc_avg, val_loss_avg = (
                val_acc_avg / len(val_loader),
                val_loss_avg / len(val_loader),
            )
        print(
            "epoch",
            epoch,
            "train_acc",
            train_acc1_avg,
            "train_loss",
            train_losses_avg,
            "val_acc",
            val_acc_avg,
            "val_loss",
            val_loss_avg,
        )
        file_save = open(savepth, mode="a")
        file_save.write(
            "\n"
            + "step:"
            + str(epoch)
            + "  loss_train:"
            + str(train_losses_avg)
            + "  acc1_train:"
            + str(train_acc1_avg)
            + "  loss_val:"
            + str(val_loss_avg)
            + "  acc1_val:"
            + str(val_acc_avg)
            + "  time_per_batch:"
            + str(time_per_batch)
        )


def worker(
    rank,
    device,
    models,
    chunks,
    output_size,
    backend,
    dist_url,
    world_size,
    epochs,
    optimizer,
    scheduler,
    len_train,
    len_val,
    warm_up,
    settings,
):
    # remember that output_size must be lists of tuple
    # forward
    torch.multiprocessing.set_sharing_strategy("file_system")
    dist.init_process_group(
        backend=backend, init_method=dist_url, world_size=world_size, rank=rank
    )
    print("process begin: ", rank)
    # train
    topk_layer = TopkLayer(settings["prun"])
    for model in models:
        model = model.to(device, non_blocking=True)
    if rank == 1 or rank == 3:
        quant_layer = RemoteQuantizationLayer(settings["quant"], 0, 0)
        dequant_layer = RemoteDeQuantizationLayer(settings["quant"], 0, 0)
    else:
        quant_layer = None
        dequant_layer = None
    for epoch in range(epochs):
        if rank == 1 or rank == 3:
            quant_layer.train()
            dequant_layer.train()
        # if rank == 3:
        #     print(models)
        for batch_iter in range(len_train):
            pass
            attention_mask = torch.rand([16, 1, 1, 128]).to(device)
            dist.recv(attention_mask, 0)  # TODO limited
            # print
            attention_mask = attention_mask.chunk(chunks)
            batches = []
            for i, model in enumerate(models):
                # model =model.to(device)
                model.train()
                batch = []

                for j in range(chunks):
                    input = torch.empty(output_size[i]).to(device)
                    # if j == 0:
                    input = input.requires_grad_()
                    if rank == 1 and settings["quant"] != 0:  # TODO limited
                        input = dequant_layer(input)
                    else:
                        input = ForwardReceive_BackwardSend.apply(
                            input, rank - 1, rank - 1, rank
                        )
                    # input = ForwardReceive_BackwardSend.apply(input,rank-1,rank-1,rank)
                    output = model(input, attention_mask[j])
                    if rank != 3:
                        output = ForwardSend_BackwardReceive.apply(
                            output, rank + 1, rank + 1, rank
                        )
                    else:
                        if settings["prun"] != 0:
                            output = topk_layer(output)
                        if settings["quant"] != 0:
                            output = quant_layer(output)
                        else:
                            output = ForwardSend_BackwardReceive.apply(
                                output, 0, 0, rank
                            )
                    # if rank != 3:
                    #     output = ForwardSend_BackwardReceive.apply(output,tuple(list(output.shape)),rank+1,rank+1,rank)
                    # else:
                    #     output = ForwardSend_BackwardReceive.apply(output,tuple(list(output.shape)),0,0,rank)
                    # print("forward","rank",rank,"part",i,"finish")
                    batch.append(output)
                    # print("rank:",rank,"part:",i,"chunk:",j,"input_size:",input.shape)
                    # print("rank",rank,"part",i,"chunk:",j,"output_size",output.shape)
                # print("rank",rank,"part",i,"length of batches",len(batch))
                batches.append(batch)
                # recv.append(batches[i][0].clone().detach())
            # forward over
            # optimizer.zero_grad()
            # print("forward over")
            # while(1):
            #     pass
            # backward
            for k in range(len(models) - 1, -1, -1):
                for l in range(chunks):

                    batches[k][l].backward(
                        torch.empty(tuple(list(batches[k][l].shape))).to(device)
                    )
                    # print("rank:",rank,"part:",k,"chunk:",l,"backward")
            optimizer.step()
            # warm_up.step()
            optimizer.zero_grad()
        if settings["warmup"] == 0:
            scheduler.step()
        else:
            scheduler.step(scheduler.last_epoch + 1)
            warm_up.dampen()
        # val
        with torch.no_grad():
            if rank == 1 or rank == 3:
                quant_layer.eval()
                dequant_layer.eval()
            for batch_iter in range(len_val):
                attention_mask = torch.rand([16, 1, 1, 128]).to(device)
                dist.recv(attention_mask, 0)  # TODO limited
                attention_mask = attention_mask.chunk(chunks)
                for i, model in enumerate(models):
                    model.eval()
                    for j in range(chunks):
                        input = torch.empty(output_size[i]).to(device)
                        if rank == 1 and settings["quant"] != 0:
                            input = dequant_layer(input)
                        else:
                            input = ForwardReceive_BackwardSend.apply(
                                input, rank - 1, rank - 1, rank
                            )
                        output = model(input, attention_mask[j])
                        if rank != 3:
                            output = ForwardSend_BackwardReceive.apply(
                                output, rank + 1, rank + 1, rank
                            )
                        else:
                            if settings["quant"] == 0:
                                output = ForwardSend_BackwardReceive.apply(
                                    output, 0, 0, rank
                                )
                            else:
                                output = quant_layer(input)


class dist_gpipe_nlp:
    def __init__(
        self,
        model_partition: List[nn.Sequential],
        devices: List[int],
        chunks: int,
        input_size: tuple,
        criterion,
        backend="nccl",
        init_method="tcp://127.0.0.1:1224",
        world_size=None,
        recv_size=None,
        save_path: str = None,
        settings=None,
    ) -> None:
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
        if save_path is not None:
            self.save_path = save_path
        if world_size == None:
            world_size = torch.cuda.device_count()
        # test divide tensor size
        self.last_device = devices[-1]
        self.chunks = chunks
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.criterion = criterion.to(devices[0])
        self.batchsize = input_size[0]
        input_size = list(input_size)
        input_size[0] = int(input_size[0] / self.chunks)
        input_size = tuple(input_size)
        input = torch.rand(input_size)
        output_size = []
        self.settings = settings
        # caclulate output size and add send recv layers
        for i, model in enumerate(model_partition):
            if i == 0 or i == len(model_partition) - 1:
                # print(input.shape)
                if i == 0:
                    input = input.type(torch.int)
                output = model(input)
                input = output
                shape = list(output.shape)
                shape = tuple(shape)
            else:
                # print(i)
                # print("input",input.shape)
                # print(model)
                output = model(input, torch.rand([4, 1, 1, 128]))
                input = output
                shape = list(output.shape)
                shape = tuple(shape)
            output_size.append(shape)

            DeferredBatchNorm.convert_deferred_batch_norm(model, self.chunks)
            # model = model.to(devices[i])
            model_partition[i] = model
        # for model in model_partition:
        #     print(model)
        # while(1):
        #     pass

        self.output_size = output_size
        # add send receive layers
        print(self.output_size)
        # partition models
        num_devices = []
        for i in devices:
            if i not in num_devices:
                num_devices.append(i)
        # print(num_devices)
        devices_index = []
        for i in num_devices:
            devices_index.append([j for j, x in enumerate(devices) if x == i])
            # count selected device's index, such as numdevice[1] is calculated in pipeline device_index[numdevice[1]]
        pass
        # print(devices_index)
        model_perdevice = []
        output_perdevice = []
        for i in num_devices:
            models = []
            outputs = []
            for j in devices_index[i]:
                # model_partition[j] = model_partition[j].to(num_devices[i])
                outputs.append(output_size[j])
                models.append(model_partition[j])
                print(next(model_partition[j].parameters()).is_cuda)
            output_perdevice.append(outputs)
            model_perdevice.append(models)
        self.model_list = model_perdevice

        self.num_devices = num_devices
        self.output_perdevice = output_perdevice
        if recv_size is not None:
            self.output_perdevice = recv_size
            pass
        print(output_perdevice)

        # self.worker = worker
        # self.worker_header = worker_header

    def session(self, train_loader, val_loader, epochs, settings: dict):
        processes = []
        for i in range(len(self.num_devices)):
            print("process start:", i)
            param_list = []
            for j, model in enumerate(self.model_list[i]):
                # param_list.append({"params": model.parameters()})
                if i == 0 and j == 1:
                    # param_list.append({"params": model.parameters(),"lr":settings["lr"]*10})
                    param_list.append({"params": model.parameters()})
                else:
                    param_list.append({"params": model.parameters()})
            if i == 0:

                optimizer = torch.optim.AdamW(
                    param_list,
                    settings["lr"],
                    # weight_decay=settings["wd"],
                )
                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10)
                # warmup_scheduler = get_scheduler(name = 'linear',optimizer = optimizer, num_warmup_steps=10*len(train_loader), num_training_steps=len(train_loader)*epochs)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs
                )
                p = mp.Process(
                    target=worker_header,
                    args=(
                        i,
                        self.num_devices[i],
                        self.model_list[i],
                        self.chunks,
                        self.criterion,
                        self.backend,
                        self.init_method,
                        self.world_size,
                        self.output_perdevice[-1],
                        epochs,
                        optimizer,
                        scheduler,
                        self.save_path,
                        warmup_scheduler,
                        train_loader,
                        val_loader,
                        settings,
                    ),
                )

            else:
                optimizer = torch.optim.AdamW(
                    param_list,
                    settings["lr"],
                    # weight_decay=settings["wd"],
                )
                warmup_schduler = warmup.LinearWarmup(optimizer, warmup_period=10)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs
                )
                # warmup_scheduler = get_scheduler(name = 'linear',optimizer = optimizer, num_warmup_steps=10*len(train_loader), num_training_steps=len(train_loader)*epochs)
                p = mp.Process(
                    target=worker,
                    args=(
                        i,
                        self.num_devices[i],
                        self.model_list[i],
                        self.chunks,
                        self.output_perdevice[i - 1],
                        self.backend,
                        self.init_method,
                        self.world_size,
                        epochs,
                        optimizer,
                        scheduler,
                        len(train_loader),
                        len(val_loader),
                        warmup_scheduler,
                        settings,
                    ),
                )
            # if i == 3:
            #     while(1):
            # pass
            p.start()

            processes.append(p)
        del train_loader, val_loader
        for process in processes:
            process.join()


# TODO 问题出现在forwardrecv，只接受了一个tensor，实际上需要两个
