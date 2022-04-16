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


# sender and recever user must input the device tensor, it can not assign the device
class FSBRFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, send_rank: int, self_rank: int, pg=None):
        ctx.recv_rank, ctx.rank, ctx.pg = send_rank, self_rank, pg
        dist.isend(input, send_rank, group=pg)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_ouput):
        recv_rank, rank, pg = ctx.recv_rank, ctx.rank, ctx.pg
        dist.recv(grad_ouput, recv_rank, group=pg)
        return grad_ouput, None, None, None


class FRBSFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, recv_rank: int, rank: int, pg=None):
        ctx.send_rank, ctx.pg = recv_rank, pg
        recv = input
        dist.recv(recv, recv_rank, group=pg)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        send_rank, pg = ctx.send_rank, ctx.pg
        send = grad_output
        dist.isend(send, send_rank, group=pg)
        return grad_output, None, None, None
