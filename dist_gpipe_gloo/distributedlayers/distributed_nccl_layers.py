"""
Author: your name
Date: 2022-04-03 11:39:13
LastEditTime: 2022-04-08 10:45:41
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe_gloo/distributedlayers/distributed_gloo_layer.py
"""
"""
Author: your name
Date: 2022-04-03 11:34:27
LastEditTime: 2022-04-03 11:37:20
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe/distributedlayers/distributed_gloo_layer.py
"""
from torch import autograd
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# used for fit the model parallel of mobilenetv2
class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class Reshape2(nn.Module):
    def __init__(self):
        super(Reshape2, self).__init__()
        pass

    def forward(self, x):
        out = x.view(x.size(0), -1)
        return out


class FSBRFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, send_rank: int, self_rank: int, pg=None):
        ctx.recv_rank, ctx.rank, ctx.pg = send_rank, self_rank, pg
        dist.isend(input, send_rank, group=pg)
        # print("forward send to",send_rank,input.shape)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_ouput):
        recv_rank, rank, pg = ctx.recv_rank, ctx.rank, ctx.pg

        dist.recv(grad_ouput, recv_rank, group=pg)
        # print(grad_ouput.get_device(),"backward recv from",recv_rank,grad_ouput.shape)
        # print()
        grad_ouput = grad_ouput.to(rank)

        return grad_ouput, None, None, None


class FRBSFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, recv_rank: int, rank: int, pg=None):
        ctx.send_rank, ctx.pg = recv_rank, pg
        # recv = input.cpu()
        recv = input
        dist.recv(recv, recv_rank, group=pg)
        # print(rank,"forward recv from", recv_rank,recv.shape)
        input = recv.to(rank)

        return input

    @staticmethod
    def backward(ctx, grad_output):
        send_rank, pg = ctx.send_rank, ctx.pg
        # send = grad_output.cpu()
        send = grad_output
        dist.isend(send, send_rank, group=pg)
        # print(grad_output.get_device(),"backward send to",send_rank)
        return grad_output, None, None, None
