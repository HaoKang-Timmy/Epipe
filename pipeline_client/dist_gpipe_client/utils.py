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
)
from torch.optim import AdamW, SGD
from transformers import get_scheduler
import torch.nn as nn
from .distributedlayers.distributed_nccl_layers import (
    FSBRFunction,
    FRBSFunction,
    FSBRFunctionClient,
    FRBSFunctionClient,
)
from .compression.compression_layer_nccl import (
    PCARecvGPU,
    PCASendGPU,
    QrecvGPU,
    QSendGPU,
    FastDequantClient,
    FastQuantClient,
    FastDequantizationServer,
    FastQuantizationServer,
    TopkPruning,
    PCARecvClient,
    PCASendClient,
    CompressionClientSend,
    QrecvClient,
    QSendClient,
)


def tensor2tuple(input: torch.tensor):
    return tuple(list(input))


# only used on cpu
def SendTensorCPU(input, settings, train_settings, chunk, edge=False):
    if train_settings["prune"] != 0:
        output = TopkPruning.apply(input, train_settings["prune"])
    elif train_settings["pca1"] != 0:
        output = PCASendClient.apply(
            input,
            train_settings["pca1"],
            settings["send_rank"],
            settings["rank"],
            settings["group_list"][chunk],
        )
    elif train_settings["sortquant"] != 0:
        output = FastQuantClient.apply(
            input,
            train_settings["quant"],
            train_settings["split"],
            settings["send_rank"],
            settings["rank"],
            settings["group_list"][chunk],
        )
    elif train_settings["quant"] != 0:
        output = QSendClient.apply(
            input,
            train_settings["quant"],
            settings["send_rank"],
            settings["device"],
            settings["group_list"][chunk],
        )
    # elif train_settings["convinsert"] != 0:
    #     output = train_settings["convlayer"](output)
    elif train_settings["poweriter1"] != 0:
        output = train_settings["poweriter1_layer"](
            input, settings["group_list"][chunk]
        )
    else:
        # print("client send",settings["send_rank"],settings["device"])
        output = FSBRFunctionClient.apply(
            input,
            settings["send_rank"],
            settings["device"],
            settings["group_list"][chunk],
        )
        # print("send over")
    return output


def SendTensor(input, settings, train_settings, chunk, edge=False):
    # server client transfer
    if settings["send_rank"] == 0 or edge is not False:
        if train_settings["prune"] != 0:
            output = TopkPruning.apply(input, train_settings["prune"])
        # if train_settings["mix"] != 0:
        #     output = CompressSendGPU.apply(
        #         input,
        #         train_settings["pca2"],
        #         settings["send_rank"],
        #         train_settings["quant"],
        #         train_settings["split"],
        #         settings["group_list"][chunk],
        #     )
        elif train_settings["pca2"] != 0:
            output = PCASendGPU.apply(
                input,
                train_settings["pca2"],
                settings["send_rank"],
                settings["rank"],
                settings["group_list"][chunk],
            )
        elif train_settings["sortquant"] != 0:
            # print("sort quant send")
            output = FastQuantizationServer.apply(
                input,
                train_settings["quant"],
                train_settings["split"],
                settings["send_rank"],
                settings["group_list"][chunk],
            )
            # print("rank:",settings["rank"],"send",settings["send_rank"])

        elif train_settings["quant"] != 0:
            output = QSendGPU.apply(
                input,
                train_settings["quant"],
                settings["send_rank"],
                settings["rank"],
                settings["group_list"][chunk],
            )
        elif train_settings["poweriter2"] != 0:
            output = train_settings["poweriter2_layer"](
                input, settings["group_list"][chunk]
            )
        else:
            output = FSBRFunction.apply(
                input,
                settings["send_rank"],
                settings["rank"],
                settings["group_list"][chunk],
            )
    else:
        # server inside transfer
        output = FSBRFunction.apply(input, settings["send_rank"], settings["rank"])
    return output


def RecvTensor(input, settings, train_settings, chunk, edge=False, time_count=False):
    # server client transfer
    if settings["recv_rank"] == 0 or edge is not False:
        # if train_settings["mix"] != 0:
        #     output = CompressRecvGPU.apply(
        #         input,
        #         train_settings["pca1"],
        #         settings["recv_rank"],
        #         train_settings["quant"],
        #         train_settings["split"],
        #         settings["group_list"][chunk],
        #     )
        if train_settings["pca1"] != 0:
            output = PCARecvGPU.apply(
                input,
                train_settings["pca1"],
                settings["recv_rank"],
                settings["rank"],
                settings["group_list"][chunk],
            )
        elif train_settings["sortquant"] != 0:
            output = FastDequantizationServer.apply(
                input,
                train_settings["quant"],
                train_settings["split"],
                settings["recv_rank"],
                settings["group_list"][chunk],
            )
            # print("rank:",settings["rank"],"recv",settings["recv_rank"])
        elif train_settings["quant"] != 0:
            output = QrecvGPU.apply(
                input,
                train_settings["quant"],
                settings["recv_rank"],
                settings["rank"],
                settings["group_list"][chunk],
            )
        elif train_settings["poweriter1"] != 0:
            output = train_settings["poweriter1_layer"](
                input, settings["group_list"][chunk]
            )
        else:
            # print("server recv",settings["recv_rank"],settings["rank"])
            output = FRBSFunction.apply(
                input,
                settings["recv_rank"],
                settings["rank"],
                settings["group_list"][chunk],
            )
    else:
        # print("rank:",settings["rank"],"recv",settings["recv_rank"])
        output = FRBSFunction.apply(input, settings["recv_rank"], settings["rank"])
    return output


def RecvTensorCPU(input, settings, train_settings, chunk, edge=False):

    if train_settings["pca2"] != 0:
        output = PCARecvClient.apply(
            input,
            train_settings["pca2"],
            settings["recv_rank"],
            settings["rank"],
            settings["group_list"][chunk],
        )
    elif train_settings["sortquant"] != 0:
        output = FastDequantClient.apply(
            input,
            train_settings["quant"],
            train_settings["split"],
            settings["recv_rank"],
            settings["rank"],
            settings["group_list"][chunk],
        )
    elif train_settings["quant"] != 0:
        output = QrecvClient.apply(
            input,
            train_settings["quant"],
            settings["recv_rank"],
            settings["device"],
            settings["group_list"][chunk],
        )
    elif train_settings["poweriter2"] != 0:
        output = train_settings["poweriter2_layer"](
            input, settings["group_list"][chunk]
        )
    else:
        # print("client recv begin")
        output = FRBSFunctionClient.apply(
            input,
            settings["recv_rank"],
            settings["device"],
            settings["group_list"][chunk],
        )
        # print("client end",settings["recv_rank"],settings["recv_rank"])
    return output


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
    args,
    model_list,
    tensor_size,
    train_loader,
    val_loader,
):
    train_settings = {}
    client_settings = {}
    client_settings["device"] = args.local_rank[
        0
    ]  # local rank to set on machine device
    client_settings["ranks"] = args.ranks  # all the ranks in global
    client_settings["rank"] = args.rank
    client_settings["backend"] = args.backend
    client_settings["dist_url"] = args.url
    client_settings["ifconfig"] = args.ifconfig
    client_settings["world_size"] = args.world_size
    client_settings["savepath"] = args.log
    client_settings["send_rank"] = args.ranks[
        1
    ]  # rank = 1,send and recv from rank 1 server
    client_settings["recv_rank"] = args.ranks[1]
    client_settings["chunks"] = args.chunks
    client_settings["showperiod"] = args.showperiod
    client_settings["recv_size"] = tensor_size[1]
    client_settings["send_size"] = tensor_size[0]
    train_settings["tasktype"] = args.tasktype
    train_settings["models"] = model_list
    train_settings["epochs"] = args.epochs
    train_settings["train_loader"] = train_loader
    train_settings["valloader"] = val_loader
    train_settings["quant"] = args.quant
    train_settings["prune"] = args.prune
    train_settings["lr"] = args.lr
    train_settings["wd"] = args.wd
    train_settings["split"] = args.split
    train_settings["sortquant"] = args.sortquant
    train_settings["pca1"] = args.pca1
    train_settings["pca2"] = args.pca2
    train_settings["poweriter1"] = args.poweriter1
    train_settings["poweriter2"] = args.poweriter2
    return client_settings, train_settings


import time


def time_count():
    torch.cuda.synchronize()
    count = time.time()
    return count
