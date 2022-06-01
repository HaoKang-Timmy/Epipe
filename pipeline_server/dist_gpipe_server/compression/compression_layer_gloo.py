"""
Author: your name
Date: 2022-04-03 11:32:55
LastEditTime: 2022-04-08 19:02:58
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe/compression/gloo_layer.py
"""

import torch.nn.functional as F
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist


def create_sparse(input: torch.tensor, bit_saving=True):
    shape = input.shape
    input = input.view(-1)
    index = input.nonzero()
    index = index.view(-1)
    if bit_saving is True:
        index = index.type(torch.bfloat16)
    src = input.index_select(0, index)
    input = input.view(shape)
    return shape, index, src


def unzip_sparse(input, index, src, shape):
    input = input.view(-1)
    input.scatter_(0, index, src)
    input = input.view(shape)
    return input


class TopkPruning(autograd.Function):
    @staticmethod
    def forward(ctx, input, ratio):
        shape = input.shape
        input = input.view(-1)
        src, index = torch.topk(torch.abs(input), int(ratio * input.shape[0]))
        mask = torch.zeros(input.shape).to(input.get_device())
        mask.index_fill_(0, index, 1.0)
        input = input * mask
        mask = mask.view(shape)
        ctx.mask = mask
        input = input.view(shape)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.mask, None, None


class TopkLayer(nn.Module):
    def __init__(self, compress_ratio, shape):
        super(TopkLayer, self).__init__()
        self.ratio = compress_ratio
        # self.mask = torch.zeros(shape)

    def forward(self, x):
        return TopkPruning.apply(x, self.ratio)


## These are gloo api
def Quantization(input: torch.tensor, bits, min_step):

    min, max = input.min(), input.max()
    step = (max - min) / (pow(2, bits) - 1)
    output = torch.round((input - min) / step - pow(2, bits - 1))
    if bits <= 8:
        output = output.type(torch.int8)
    elif bits <= 16:
        output = output.type(torch.int16)
    else:
        output = output.type(torch.int32)

    min_step[0] = min.item()
    min_step[1] = step.item()
    return min_step, output


def Dequantization(input: torch.tensor, bits, min_step):
    input = input.type(torch.float32)

    output = (input + pow(2, bits - 1)) * min_step[1] + min_step[0]
    output = output.requires_grad_()
    return output


def QtensorSendonCPU(min_step, output, send_rank):
    min_step_cpu = min_step.cpu()
    output_cpu = output.cpu()
    dist.isend(min_step_cpu, send_rank)
    dist.isend(output_cpu, send_rank)


def QtensorRecvonCPU(min_step, input, bits, rank):
    """
    return min_step and output the same device as the input
    """
    device = input.get_device()
    min_step_cpu = min_step.cpu()
    input_cpu = input.cpu()
    if bits <= 8:
        input_cpu = input_cpu.type(torch.int8)
    elif bits <= 16:
        input_cpu = input_cpu.type(torch.int16)
    else:
        input_cpu = input_cpu.type(torch.int32)
    # print("prepare recv",min_step_cpu,input_cpu)
    dist.recv(min_step_cpu, rank)
    dist.recv(input_cpu, rank)
    if device >= 0:
        min_step = min_step_cpu.to(device)
        input = input_cpu.to(device)
    else:
        min_step = min_step_cpu
        input = input_cpu
    # print("Qrecv")
    # print("recv")
    return min_step, input


class Qsend(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, send_rank):
        ctx.bits, ctx.recv_rank = (bits, send_rank)
        if input.get_device() >= 0:
            min_step = torch.zeros([2]).to(input.get_device())
        else:
            min_step = torch.zeros([2])
        min_step, output = Quantization(input, bits, min_step)
        ctx.recv = output
        QtensorSendonCPU(min_step, output, send_rank)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, recv_rank = (ctx.bits, ctx.recv_rank)
        if grad_output.get_device() >= 0:
            min_step = torch.zeros([2]).to(grad_output.get_device())
        else:
            min_step = torch.zeros([2])
        recv = ctx.recv
        min_step, input = QtensorRecvonCPU(min_step, recv, bits, recv_rank)
        grad_output = Dequantization(input, bits, min_step[0], min_step[1])
        # print(grad_output)
        return grad_output, None, None, None, None


# class QSendLayerGloo(nn.Module):
#     def __init__(self, bits, send_rank, rank, sparse=False) -> None:
#         super(QSendLayerGloo, self).__init__()
#         self.bits = bits
#         self.min_step = torch.tensor([0.0, 0.0])
#         self.send_rank = send_rank
#         self.rank = rank
#         self.sparse = sparse

#     def forward(self, input):

#         return QSend.apply(input, self.bits, self.min_step, self.send_rank, self.rank)


class Qrecv(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, recv_rank):
        ctx.bits, ctx.send_rank = (
            bits,
            recv_rank,
        )
        if input.get_device() >= 0:
            min_step = torch.zeros([2]).to(input.get_device())
        else:
            min_step = torch.zeros([2])
        min_step, recv = QtensorRecvonCPU(min_step, input, bits, recv_rank)
        # print("recv")

        input = Dequantization(recv, bits, min_step[0], min_step[1])
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, send_rank = (ctx.bits, ctx.send_rank)
        if grad_output.get_device() >= 0:
            min_step = torch.zeros([2]).to(grad_output.get_device())
        else:
            min_step = torch.zeros([2])
        min_step_cpu, output = Quantization(grad_output, bits, min_step)
        # dist.send(min_step.cpu(), recv_rank)
        # dist.send(output.cpu(), recv_rank)
        QtensorSendonCPU(min_step_cpu, output, send_rank)
        return grad_output, None, None, None, None


# class QRecvLayerGloo(nn.Module):
#     def __init__(self, bits, recv_rank, rank) -> None:
#         super(QRecvLayerGloo, self).__init__()
#         self.bits = bits
#         self.recv_rank = recv_rank
#         self.rank = rank
#         self.min_step = torch.tensor([0.0, 0.0])

#     def forward(self, input):
#         return Qrecv.apply(input, self.bits, self.min_step, self.recv_rank, self.rank)
