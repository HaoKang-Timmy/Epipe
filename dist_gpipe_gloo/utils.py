import torchvision.transforms as transforms
import torchvision
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import torch
import torch.distributed as dist
from .compression import (
    TopkLayer,
    QSendLayerGPU,
    QRecvLayerGPU,
    SortQuantGPU,
    SortDeQuantGPU,
)
from torch.optim import AdamW, SGD
from transformers import get_scheduler
import torch.nn as nn
from .distributedlayers.distributed_nccl_layers import FSBRFunction, FRBSFunction


def tensor2tuple(input: torch.tensor):
    return tuple(list(input))


def SendTensor(
    input,
    send_rank,
    self_rank,
    prune_layer=None,
    quant_layer=None,
    send_layer=None,
    sortquant=None,
    split=0,
    bits=0,
):
    if prune_layer is not None:
        input = prune_layer(input)
    if quant_layer is not None:
        input = quant_layer(input)
    elif send_layer is not None:
        input = FSBRFunction.apply(input, send_rank, self_rank)
    elif sortquant is not None:
        input = SortQuantGPU.apply(input, bits, split, send_rank)
    return input


def RecvTensor(
    input,
    recv_rank,
    self_rank,
    dequant_layer=None,
    recv_layer=None,
    sortdequant=None,
    split=0,
    bits=0,
):
    if dequant_layer is not None:
        input = dequant_layer(input)
    elif recv_layer is not None:
        input = FRBSFunction.apply(input, recv_rank, self_rank)
        # print(input)
    elif sortdequant is not None:
        input = SortDeQuantGPU.apply(input, bits, split, recv_rank)
    return input


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def make_dictions(
    client_train_settings,
    client_settings,
    server_train_settings_list,
    server_settings_list,
    args,
    model_list,
    devices,
    tensor_size,
    train_loader,
    val_loader,
):
    client_settings["device"] = 0

    client_settings["rank"] = devices[0]
    client_settings["backend"] = args.bachend
    client_settings["dist_url"] = args.url
    client_settings["world_size"] = args.world_size
    client_settings["savepath"] = args.log
    client_settings["send_rank"] = devices[1]
    client_settings["recv_rank"] = devices[-1]
    client_settings["chunks"] = args.chunks
    client_settings["showperiod"] = args.showperiod
    client_settings["recv_size"] = tensor_size[0][1]
    client_settings["send_size"] = tensor_size[0][0]
    client_train_settings["tasktype"] = args.tasktype
    client_train_settings["models"] = model_list[client_settings["rank"]]
    client_train_settings["epochs"] = args.epochs
    client_train_settings["train_loader"] = train_loader
    client_train_settings["valloader"] = val_loader
    client_train_settings["quant"] = args.quant
    client_train_settings["prune"] = args.prune
    client_train_settings["lr"] = args.lr
    client_train_settings["wd"] = args.wd
    client_train_settings["split"] = args.split
    client_train_settings["sortquant"] = args.sortquant
    for server_num in range(len(devices) - 1):
        train_settings = {}
        server_settings = {}
        server_settings["device"] = devices[server_num + 1]
        server_settings["rank"] = server_num + 1
        server_settings["backend"] = args.bachend
        server_settings["dist_url"] = args.url
        server_settings["world_size"] = args.world_size
        server_settings["send_size"] = tensor_size[server_num + 1][0]
        server_settings["recv_size"] = tensor_size[server_num + 1][1]
        if server_num != len(devices) - 2:
            server_settings["send_rank"] = server_num + 2
        else:
            server_settings["send_rank"] = client_settings["rank"] = devices[0]
        server_settings["recv_rank"] = server_num
        server_settings["chunks"] = args.chunks
        train_settings["epochs"] = args.epochs
        train_settings["tasktype"] = args.tasktype
        train_settings["models"] = model_list[server_settings["rank"]]
        train_settings["device"] = devices[server_num + 1]
        train_settings["lr"] = args.lr
        train_settings["wd"] = args.wd
        train_settings["split"] = args.split
        train_settings["sortquant"] = args.sortquant
        train_settings["prune"] = args.prune
        train_settings["quant"] = args.quant
        train_settings["len_trainloader"] = len(train_loader)
        train_settings["len_valloader"] = len(val_loader)
        server_settings_list.append(server_settings)
        server_train_settings_list.append(train_settings)
