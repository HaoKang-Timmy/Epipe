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
def Quantization2CPU(input: torch.tensor, bits, min_step):

    min, max = input.min(), input.max()
    step = (max - min) / (pow(2, bits) - 1)
    output = torch.round((input - min) / step - pow(2, bits - 1))
    output = output.cpu()
    if bits <= 8:
        output = output.type(torch.int8)
    elif bits <= 16:
        output = output.type(torch.int16)
    else:
        output = output.type(torch.int32)

    min_step[0] = min.item()
    min_step[1] = step.item()
    min_step_cpu = min_step.cpu()
    return min_step_cpu, output


def DequantizationonGPU(input: torch.tensor, bits, min, step):
    input = input.type(torch.cuda.FloatTensor)

    output = (input + pow(2, bits - 1)) * step + min
    output = output.requires_grad_()
    return output


def QtensorSendonCPU(min_step, output, send_rank):
    handle1 = dist.send(min_step, send_rank)
    handle2 = dist.send(output, send_rank)
    # handle2.wait()
    # return handle2
    # print("send",min_step,output)
    #TODO gloo send has bug



def QtensorRecvonCPU(min_step, input, bits, rank):
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
    print("Qrecv")
    # print("recv")
    return min_step_cpu, input_cpu


class QSend(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, min_step, send_rank, rank):
        ctx.bits, ctx.recv_rank, ctx.rank, ctx.min_step = (
            bits,
            send_rank,
            rank,
            min_step,
        )
        min_step_cpu, output = Quantization2CPU(input, bits, min_step)
        QtensorSendonCPU(min_step_cpu, output, send_rank)
        # handle.wait()
        print("Qsend")
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, recv_rank, rank, min_step = (
            ctx.bits,
            ctx.recv_rank,
            ctx.rank,
            ctx.min_step,
        )
        min_step_cpu, input_cpu = QtensorRecvonCPU(min_step, input, bits, recv_rank)
        input = input_cpu.to(rank)
        min_step = min_step_cpu.to(rank)
        grad_output = DequantizationonGPU(input, bits, min_step[0], min_step[1])
        # print(grad_output)
        return grad_output, None, None, None, None




class QSendLayerGloo(nn.Module):
    def __init__(self, bits, send_rank, rank, sparse=False) -> None:
        super(QSendLayerGloo, self).__init__()
        self.bits = bits
        self.min_step = torch.tensor([0.0, 0.0])
        self.send_rank = send_rank
        self.rank = rank
        self.sparse = sparse

    def forward(self, input):

        return QSend.apply(
            input, self.bits, self.min_step, self.send_rank, self.rank
        )



class Qrecv(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, min_step, recv_rank, rank):
        ctx.bits, ctx.send_rank, ctx.min_step = (
            bits,
            recv_rank,
            min_step,
        )
        min_step_cpu,recv = QtensorRecvonCPU(min_step,input,bits,recv_rank)
        min_step = min_step_cpu.to(rank)
        recv = recv.to(rank)
        # print("recv")
        input = DequantizationonGPU(recv, bits, min_step[0], min_step[1])
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, send_rank,  min_step = (
            ctx.bits,
            ctx.send_rank,
            ctx.min_step,
        )
        min_step_cpu, output = Quantization2CPU(grad_output, bits, min_step)
        # dist.send(min_step.cpu(), recv_rank)
        # dist.send(output.cpu(), recv_rank)
        QtensorSendonCPU(min_step_cpu,output,send_rank)
        return grad_output, None, None, None, None



class QRecvLayerGloo(nn.Module):
    def __init__(self, bits, recv_rank, rank) -> None:
        super(QRecvLayerGloo, self).__init__()
        self.bits = bits
        self.recv_rank = recv_rank
        self.rank = rank
        self.min_step = torch.tensor([0.0, 0.0])

    def forward(self, input):
        return Qrecv.apply(input, self.bits, self.min_step, self.recv_rank, self.rank)
