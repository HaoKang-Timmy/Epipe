"""
Author: your name
Date: 2022-04-03 11:32:55
LastEditTime: 2022-04-08 19:02:58
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe/compression/gloo_layer.py
"""

from lib2to3.pgen2 import pgen
from numpy import dtype
import torch.nn.functional as F
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist
import time
from functions import *


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
        if input.get_device() >= 0:
            mask = torch.zeros(input.shape).to(input.get_device())
        else:
            mask = torch.zeros(input.shape)
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


class QSendGPU(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, send_rank, rank, pg=None):
        ctx.bits, ctx.recv_rank, ctx.rank, ctx.pg = (bits, send_rank, rank, pg)
        (
            output,
            min,
            step,
        ) = QuantizationGPU(input, bits)
        # print("quant send",output)
        ctx.min, ctx.step = min, step
        ctx.input = output
        # print("pre send to",send_rank )
        QtensorSend(output, min, step, send_rank, pg)
        # print("send")

        # print("input",input.shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, recv_rank, rank, min, step, pg = (
            ctx.bits,
            ctx.recv_rank,
            ctx.rank,
            ctx.min,
            ctx.step,
            ctx.pg,
        )
        input = ctx.input

        min, step, input = QtensorRecvonGPU1(input, bits, min, step, recv_rank, pg)
        # print("backward recv",input,min,step)
        if bits <= 16 and bits > 8:
            input = input.view(dtype=torch.int16)

        grad_output = Dequantizationon(input, bits, min, step)
        # print(grad_output)
        # print(grad_output.shape)
        return grad_output, None, None, None, None


class QSendLayerGPU(nn.Module):
    def __init__(self, bits, send_rank, rank, pg_group, sparse=False) -> None:
        super(QSendLayerGPU, self).__init__()
        self.bits = bits
        self.send_rank = send_rank
        self.rank = rank
        self.sparse = sparse
        self.pg_group = pg_group

    def forward(self, input):

        return QSendGPU.apply(input, self.bits, self.send_rank, self.rank)


def int32to10():
    pass


def int32to12():
    pass


def QtensorRecvonGPU(input, bits, min, step, recv_rank, pg=None):
    # print(input.shape)
    if bits <= 8:
        input = input.type(torch.int8)
    # elif bits == 10:
    #     input = int32to10()TODO
    # elif bits == 12:
    #     input = int32to12()
    elif bits <= 16:
        input = input.type(torch.int16)
        input = input.view(dtype=torch.int8)
    dist.recv(min, recv_rank, group=pg)
    dist.recv(step, recv_rank, group=pg)
    dist.recv(input, recv_rank, group=pg)
    return min, step, input


class QrecvGPU(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, recv_rank, rank, pg=None):
        min = torch.tensor(0.0).to(rank)
        step = torch.tensor(0.0).to(rank)
        ctx.bits, ctx.send_rank, ctx.min, ctx.step, ctx.pg = (
            bits,
            recv_rank,
            min,
            step,
            pg,
        )
        min, step, recv = QtensorRecvonGPU(input, bits, min, step, recv_rank, pg)
        # print("recv quant",recv)
        if bits <= 16 and bits > 8:
            recv = recv.view(dtype=torch.int16)
        input = Dequantizationon(recv, bits, min, step)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, send_rank, min, step, pg = (
            ctx.bits,
            ctx.send_rank,
            ctx.min,
            ctx.step,
            ctx.pg,
        )
        output, min, step = QuantizationGPU(grad_output, bits)
        # dist.send(min_step.cpu(), recv_rank)
        # dist.send(output.cpu(), recv_rank)
        QtensorSend(output, min, step, send_rank, pg)
        return grad_output, None, None, None, None


class QRecvLayerGPU(nn.Module):
    def __init__(self, bits, recv_rank, rank, pg_group) -> None:
        super(QRecvLayerGPU, self).__init__()
        self.bits = bits
        self.recv_rank = recv_rank
        self.rank = rank
        self.min_step = torch.tensor([0.0, 0.0])
        self.pg_group = pg_group

    def forward(self, input):
        return QrecvGPU.apply(input, self.bits, self.recv_rank, self.rank)


class QSendClient(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, send_rank, device, pg=None):
        ctx.bits, ctx.recv_rank, ctx.device, ctx.pg = (bits, send_rank, device, pg)
        min = torch.tensor(0.0)
        step = torch.tensor(0.0)
        output, min, step = QuantizationCPU(input, bits)
        # print("quant send",output)

        # print("pre send to",send_rank )

        # print("send")
        min = min.to(device)
        step = step.to(device)
        output = output.to(device)
        ctx.min, ctx.step = min, step
        ctx.input = output
        dist.send(min, send_rank, group=pg)
        dist.send(step, send_rank, group=pg)
        dist.send(output, send_rank, group=pg)
        # print("input",input.shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, recv_rank, device, min, step, pg = (
            ctx.bits,
            ctx.recv_rank,
            ctx.device,
            ctx.min,
            ctx.step,
            ctx.pg,
        )
        input = ctx.input
        # input = input.to(device)
        min, step, input = QtensorRecvonGPU1(input, bits, min, step, recv_rank, pg)
        # print("backward recv",input,min,step)
        min = min.to("cpu")
        step = step.to("cpu")
        input = input.to("cpu")
        if bits <= 16 and bits > 8:
            input = input.view(dtype=torch.int16)

        grad_output = Dequantizationon(input, bits, min, step)
        return grad_output, None, None, None, None


class QrecvClient(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, recv_rank, device, pg=None):
        min = torch.tensor(0.0).to(device)
        step = torch.tensor(0.0).to(device)
        ctx.bits, ctx.send_rank, ctx.device, ctx.min, ctx.step, ctx.pg = (
            bits,
            recv_rank,
            device,
            min,
            step,
            pg,
        )
        input = input.to(device)

        min, step, recv = QtensorRecvonGPU(input, bits, min, step, recv_rank, pg)
        # print("recv quant",recv)
        if bits <= 16 and bits > 8:
            recv = recv.view(dtype=torch.int16)
        recv = recv.to("cpu")
        min = min.to("cpu")
        step = step.to("cpu")

        input = Dequantizationon(recv, bits, min, step)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, send_rank, device, min, step, pg = (
            ctx.bits,
            ctx.send_rank,
            ctx.device,
            ctx.min,
            ctx.step,
            ctx.pg,
        )
        output, min, step = QuantizationCPU(grad_output, bits)
        # dist.send(min_step.cpu(), recv_rank)
        # dist.send(output.cpu(), recv_rank)
        min = min.to(device)
        step = step.to(device)
        output = output.to(device)

        QtensorSend(output, min, step, send_rank, pg)
        return grad_output, None, None, None, None


# need to add each number of split to get and dequantize
class FastQuantClient(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, split_bits, send_rank, device, pg=None):
        ctx.recv_rank, ctx.bits, ctx.split_bits, ctx.pg = (
            send_rank,
            bits,
            split_bits,
            pg,
        )
        ctx.device = device
        shape = input.shape
        min_step = torch.zeros([2**split_bits, 2])
        # min_step, output = SortQuantization(input, bits, split_bits, min_step)
        min_step, output = FastQuantization(input, bits, split_bits, min_step)
        min_step = min_step.to(device)
        output = output.to(device)
        dist.isend(min_step, send_rank, group=pg)
        dist.isend(output, send_rank, group=pg)
        input = input.view(shape)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        recv_rank, bits, split_bits, pg = (
            ctx.recv_rank,
            ctx.bits,
            ctx.split_bits,
            ctx.pg,
        )
        device = ctx.device
        shape = grad_output.shape
        grad_output = grad_output.view(-1)
        # must use the right dtype to recv tensor
        if bits + split_bits <= 8:
            recv = grad_output.type(torch.int8)
        else:
            recv = grad_output.type(torch.int16)
            recv = recv.view(torch.int8)

        min_step = torch.zeros([2**split_bits, 2])
        min_step = min_step.to(device)
        recv = recv.to(device)
        dist.recv(min_step, recv_rank, group=pg)
        dist.recv(recv, recv_rank, group=pg)
        min_step = min_step.cpu()
        recv = recv.cpu()
        grad_output = FastDequantization(recv, bits, split_bits, min_step, grad_output)
        grad_output = grad_output.view(shape)
        # print(grad_output)
        return grad_output, None, None, None, None, None


# no sparse
class FastQuantizationServer(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, split_bits, send_rank, pg=None):
        ctx.recv_rank, ctx.bits, ctx.split_bits, ctx.pg = (
            send_rank,
            bits,
            split_bits,
            pg,
        )
        shape = input.shape
        min_step = torch.zeros([2**split_bits, 2]).to(input.get_device())
        min_step, output = SortQuantization(input, bits, split_bits, min_step)
        # print("quant min step",min_step)
        dist.isend(min_step, send_rank, group=pg)
        dist.isend(output, send_rank, group=pg)
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        recv_rank, bits, split_bits, pg = (
            ctx.recv_rank,
            ctx.bits,
            ctx.split_bits,
            ctx.pg,
        )
        shape = grad_output.shape
        grad_output = grad_output.view(-1)
        # must use the right dtype to recv tensor
        if bits + split_bits <= 8:
            recv = grad_output.type(torch.int8)
        else:
            recv = grad_output.type(torch.int16)
            recv = recv.view(torch.int8)

        min_step = torch.zeros([2**split_bits, 2]).to(grad_output.get_device())
        dist.recv(min_step, recv_rank, group=pg)
        dist.recv(recv, recv_rank, group=pg)
        # should view to int16 since we represent int16 with int8
        # lower_bound = min_step[:, -2].to("cpu").type(torch.int32)
        # upper_bound = min_step[:, -1].to("cpu").type(torch.int32)
        grad_output = FastDequantization(recv, bits, split_bits, min_step, grad_output)
        grad_output = grad_output.view(shape)
        # print("server backward recv")
        # print(grad_output)
        return grad_output, None, None, None, None


class FastDequantizationServer(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, split_bits, recv_rank, pg=None):
        ctx.send_rank, ctx.bits, ctx.split_bits, ctx.pg = (
            recv_rank,
            bits,
            split_bits,
            pg,
        )
        shape = input.shape
        # print(shape)
        input = input.view(-1)
        if bits + split_bits <= 8:
            recv = input.type(torch.int8)
        else:
            recv = input.type(torch.int16)
            # print(recv.shape)
            recv = recv.view(torch.int8)
        #     print(recv.shape)
        # print("dequant recv",recv.shape)
        min_step = torch.zeros([2**split_bits, 2]).to(input.get_device())
        # recv = torch.rand([64,32,112,112]).to(input.get_device())
        # TODO change the recving method

        dist.recv(min_step, recv_rank, group=pg)
        dist.recv(recv, recv_rank, group=pg)

        # input = FastDequantizationCPU(input,bits,split_bits,min_step,recv)
        # print(end - start)
        input = FastDequantization(recv, bits, split_bits, min_step, input)
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        send_rank, bits, split_bits, pg = (
            ctx.send_rank,
            ctx.bits,
            ctx.split_bits,
            ctx.pg,
        )
        # torch.cuda.synchronize()
        # start = time.time()
        shape = grad_output.shape
        # print(grad_output)

        min_step = torch.zeros([2**split_bits, 2]).to(grad_output.get_device())
        min_step, output = SortQuantization(grad_output, bits, split_bits, min_step)

        dist.isend(min_step, send_rank, group=pg)
        dist.isend(output, send_rank, group=pg)
        grad_output = grad_output.view(shape)
        # print(grad_output)
        # torch.cuda.synchronize()
        # print(time.time()- start)
        return grad_output, None, None, None, None, None


class FastDequantClient(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, split_bits, recv_rank, device, pg=None):
        ctx.send_rank, ctx.bits, ctx.split_bits, ctx.pg = (
            recv_rank,
            bits,
            split_bits,
            pg,
        )
        ctx.device = device
        shape = input.shape
        input = input.view(-1)
        if bits + split_bits <= 8:
            recv = input.type(torch.int8)
        else:
            recv = input.type(torch.int16)
            recv = recv.view(torch.int8)

        min_step = torch.zeros([2**split_bits, 2])
        min_step = min_step.to(device)
        recv = recv.to(device)
        dist.recv(min_step, recv_rank, group=pg)
        dist.recv(recv, recv_rank, group=pg)
        # print('recv min step', min_step)
        min_step = min_step.cpu()
        recv = recv.cpu()
        input = FastDequantization(recv, bits, split_bits, min_step, input)
        # input = input * 1.0
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        send_rank, bits, split_bits, pg = (
            ctx.send_rank,
            ctx.bits,
            ctx.split_bits,
            ctx.pg,
        )
        device = ctx.device
        shape = grad_output.shape
        min_step = torch.zeros([2**split_bits, 2])
        # min_step, output = SortQuantization(grad_output, bits, split_bits, min_step)
        min_step, output = FastQuantization(grad_output, bits, split_bits, min_step)
        min_step = min_step.to(device)
        output = output.to(device)
        dist.isend(min_step, send_rank, group=pg)
        dist.isend(output, send_rank, group=pg)
        grad_output = grad_output.view(shape)
        # print(output)
        # print(min_step)
        # print(grad_output.shape)
        # print("server backward send")
        return grad_output, None, None, None, None, None


class PCASendClient(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, send_rank, device, pg=None):
        ctx.q, ctx.recv_rank, ctx.device, ctx.pg = q, send_rank, device, pg
        U, S, V = torch.svd_lowrank(input, q=q)
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        # print("send U",U.shape)
        # print("send S",S.shape)
        # print("send V",V.shape)
        dist.isend(U, send_rank, group=pg)
        dist.isend(S, send_rank, group=pg)
        dist.isend(V, send_rank, group=pg)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        q, recv_rank, device, pg = ctx.q, ctx.recv_rank, ctx.device, ctx.pg
        shape = list(grad_output.shape)
        v_shape = shape.copy()
        v_shape[-2] = shape[-1]
        v_shape[-1] = q
        shape[-1] = q
        s_shape = shape[:-1]
        s_shape[-1] = q
        U = torch.empty(shape).to(device)
        V = torch.empty(v_shape).to(device)
        S = torch.empty(s_shape).to(device)
        dist.recv(U, recv_rank, group=pg)
        dist.recv(S, recv_rank, group=pg)
        dist.recv(V, recv_rank, group=pg)
        U = U.cpu()
        V = V.cpu()
        S = S.cpu()
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        grad_output = torch.matmul(output[..., :, :], V[..., :, :])
        return grad_output, None, None, None, None


class PCARecvClient(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, recv_rank, device, pg=None):
        ctx.q, ctx.send_rank, ctx.device, ctx.pg = q, recv_rank, device, pg
        shape = list(input.shape)
        v_shape = shape.copy()
        v_shape[-2] = shape[-1]
        v_shape[-1] = q
        shape[-1] = q
        s_shape = shape[:-1]
        s_shape[-1] = q
        U = torch.empty(shape).to(device)
        V = torch.empty(v_shape).to(device)
        S = torch.empty(s_shape).to(device)
        dist.recv(U, recv_rank, group=pg)
        dist.recv(S, recv_rank, group=pg)
        dist.recv(V, recv_rank, group=pg)
        U = U.cpu()
        V = V.cpu()
        S = S.cpu()
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        input = torch.matmul(output[..., :, :], V[..., :, :])
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        q, recv_rank, device, pg = ctx.q, ctx.send_rank, ctx.device, ctx.pg
        U, S, V = torch.svd_lowrank(grad_output, q=q)
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        dist.isend(U, recv_rank, group=pg)
        dist.isend(S, recv_rank, group=pg)
        dist.isend(V, recv_rank, group=pg)
        return grad_output, None, None, None, None


class PCASendGPU(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, send_rank, device, pg=None):
        ctx.q, ctx.recv_rank, ctx.device, ctx.pg = q, send_rank, device, pg
        input = input.cpu()
        U, S, V = torch.svd_lowrank(input, q=q)
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        dist.isend(U, send_rank, group=pg)
        dist.isend(S, send_rank, group=pg)
        dist.isend(V, send_rank, group=pg)
        input = input.to(device)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        q, recv_rank, device, pg = ctx.q, ctx.recv_rank, ctx.device, ctx.pg
        shape = list(grad_output.shape)
        v_shape = shape.copy()
        v_shape[-2] = shape[-1]
        v_shape[-1] = q
        shape[-1] = q
        s_shape = shape[:-1]
        s_shape[-1] = q
        U = torch.empty(shape).to(device)
        V = torch.empty(v_shape).to(device)
        S = torch.empty(s_shape).to(device)
        dist.recv(U, recv_rank, group=pg)
        dist.recv(S, recv_rank, group=pg)
        dist.recv(V, recv_rank, group=pg)

        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        grad_output = torch.matmul(output[..., :, :], V[..., :, :])
        return grad_output, None, None, None, None


class PCARecvGPU(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, recv_rank, device, pg=None):
        ctx.q, ctx.send_rank, ctx.device, ctx.pg = q, recv_rank, device, pg
        shape = list(input.shape)
        # print("before",shape)
        v_shape = shape.copy()
        v_shape[-2] = shape[-1]
        v_shape[-1] = q
        shape[-1] = q
        s_shape = shape[:-1]
        s_shape[-1] = q
        # print("after",shape)
        U = torch.empty(shape).to(device)
        V = torch.empty(v_shape).to(device)
        S = torch.empty(s_shape).to(device)
        # print("recv U",U.shape)
        # print("recv S",S.shape)
        # print("recv V",V.shape)
        dist.recv(U, recv_rank, group=pg)
        dist.recv(S, recv_rank, group=pg)
        dist.recv(V, recv_rank, group=pg)

        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        # print("U",U.shape)
        # print("S",S.shape)
        # print("V",V.shape)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        input = torch.matmul(output[..., :, :], V[..., :, :])
        # print("server recv2",input.shape)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        q, send_rank, device, pg = ctx.q, ctx.send_rank, ctx.device, ctx.pg
        output = grad_output.cpu()
        U, S, V = torch.svd_lowrank(output, q=q)
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        dist.isend(U, send_rank, group=pg)
        dist.isend(S, send_rank, group=pg)
        dist.isend(V, send_rank, group=pg)
        return grad_output, None, None, None, None


class CompressionClientSend(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, send_rank, device, bits, split_bits, pg=None):
        ctx.bits, ctx.split_bits, ctx.device, ctx.pg = bits, split_bits, device, pg
        ctx.recv_rank = send_rank
        # print(input)
        U, S, V = torch.svd_lowrank(input, q=q)

        #
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        #
        dist.isend(U, send_rank, group=pg)
        dist.isend(S, send_rank, group=pg)
        dist.isend(V, send_rank, group=pg)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        bits, split_bits, device, pg = ctx.bits, ctx.split_bits, ctx.device, ctx.pg
        shape = grad_output.shape
        recv_rank = ctx.recv_rank
        grad_output = grad_output.view(-1)
        if bits + split_bits <= 8:
            recv = grad_output.type(torch.int8)
        else:
            recv = grad_output.type(torch.int16)
        recv = recv.view(torch.int8)
        min_step = torch.zeros([2**split_bits, 2]).to(device)
        recv = recv.to(device)
        dist.recv(min_step, recv_rank, group=pg)
        dist.recv(recv, recv_rank, group=pg)
        min_step = min_step.cpu()
        recv = recv.cpu()
        # start = time.time()
        FastDeQuantization(recv, bits, split_bits, min_step, grad_output)
        # print(time.time() - start)
        grad_output = grad_output.view(shape)
        return grad_output, None, None, None, None, None, None


class CompressionClientRecv(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, recv_rank, device, bits, split_bits, pg=None):
        ctx.send_rank, ctx.bits, ctx.split_bits, ctx.pg = (
            recv_rank,
            bits,
            split_bits,
            pg,
        )
        ctx.q = q
        ctx.device = device
        shape = input.shape
        input = input.view(-1)
        if bits + split_bits <= 8:
            recv = input.type(torch.int8)
        else:
            recv = input.type(torch.int16)
            recv = recv.view(torch.int8)

        min_step = torch.zeros([2**split_bits, 2])
        min_step = min_step.to(device)
        recv = recv.to(device)
        dist.recv(min_step, recv_rank, group=pg)
        dist.recv(recv, recv_rank, group=pg)
        min_step = min_step.cpu()
        recv = recv.cpu()
        input = FastDeQuantization(recv, bits, split_bits, min_step, input)
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        q, send_rank, device, pg = ctx.q, ctx.send_rank, ctx.device, ctx.pg
        U, S, V = torch.svd_lowrank(grad_output, q=q)
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        dist.isend(U, send_rank, group=pg)
        dist.isend(S, send_rank, group=pg)
        dist.isend(V, send_rank, group=pg)
        return grad_output, None, None, None, None, None, None


class CompressRecvGPU(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, recv_rank, bits, split_bits, pg=None):
        ctx.bits, ctx.split_bits, ctx.pg = bits, split_bits, pg
        ctx.send_rank = recv_rank
        ctx.min_step = torch.zeros([2**split_bits, 2]).to(input.get_device())
        shape = list(input.shape)
        v_shape = shape
        v_shape[-2] = shape[-1]
        v_shape[-1] = q
        shape[-1] = q
        s_shape = shape[:-1]
        s_shape[-1] = q
        U = torch.empty(shape).to(input.get_device())
        V = torch.empty(v_shape).to(input.get_device())
        S = torch.empty(s_shape).to(input.get_device())
        dist.recv(U, recv_rank, group=pg)
        dist.recv(S, recv_rank, group=pg)
        dist.recv(V, recv_rank, group=pg)
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)

        output = torch.matmul(U[..., :, :], S[..., :, :])
        input = torch.matmul(output[..., :, :], V[..., :, :])
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        send_rank, bits, split_bits, pg = (
            ctx.send_rank,
            ctx.bits,
            ctx.split_bits,
            ctx.pg,
        )
        shape = grad_output.shape

        # some = time.time()
        min_step = ctx.min_step
        # print(time.time() - some)

        # some = time.time()
        min_step, output = SortQuantization(grad_output, bits, split_bits, min_step)
        # print(time.time() - some)
        dist.isend(min_step, send_rank, group=pg)
        dist.isend(output, send_rank, group=pg)
        grad_output = grad_output.view(shape)
        return grad_output, None, None, None, None, None, None


class CompressSendGPU(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, send_rank, bits, split_bits, pg=None):
        ctx.q, ctx.recv_rank, ctx.pg = q, send_rank, pg
        shape = input.shape
        min_step = torch.zeros([2**split_bits, 2]).to(input.get_device())
        min_step, output = SortQuantization(input, bits, split_bits, min_step)
        dist.isend(min_step, send_rank, group=pg)
        dist.isend(output, send_rank, group=pg)
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        q, recv_rank, pg = ctx.q, ctx.recv_rank, ctx.pg
        shape = list(grad_output.shape)
        shape[-1] = q
        s_shape = shape[:-1]
        s_shape[-1] = q
        U = torch.empty(shape).to(grad_output.get_device())
        V = torch.empty(shape).to(grad_output.get_device())
        S = torch.empty(s_shape).to(grad_output.get_device())
        dist.recv(U, recv_rank, group=pg)
        dist.recv(S, recv_rank, group=pg)
        dist.recv(V, recv_rank, group=pg)
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        grad_output = torch.matmul(output[..., :, :], V[..., :, :])
        return grad_output, None, None, None, None, None


class PowerSVDSendClient(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        p_buffer,
        q_buffer,
        n_iter,
        grad_p_buffer,
        grad_q_buffer,
        device,
        send_rank,
        pg=None,
    ):
        ctx.grad_p_buffer, ctx.grad_q_buffer = grad_p_buffer, grad_q_buffer
        ctx.recv_rank, ctx.pg, ctx.device = send_rank, pg, device
        p, q = PowerSVD(input, q_buffer, p_buffer, n_iter)
        p = p.to(device)
        q = q.to(device)
        dist.isend(p, send_rank, group=pg)
        dist.isend(q, send_rank, group=pg)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        p_buffer, q_buffer = ctx.grad_p_buffer, ctx.grad_q_buffer
        recv_rank, pg, device = ctx.recv_rank, ctx.pg, ctx.device
        p_buffer[0] = p_buffer[0].to(device)
        q_buffer[0] = q_buffer[0].to(device)
        dist.recv(p_buffer[0], recv_rank, group=pg)
        dist.recv(q_buffer[0], recv_rank, group=pg)
        p_buffer[0] = p_buffer[0].to("cpu")
        q_buffer[0] = q_buffer[0].to("cpu")
        grad_output = PowerSVDDecompress(p_buffer[0], q_buffer[0], grad_output.shape)
        return grad_output, None, None, None, None, None, None, None, None


class PowerSVDRecvClient(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        p_buffer,
        q_buffer,
        n_iter,
        grad_p_buffer,
        grad_q_buffer,
        device,
        recv_rank,
        pg=None,
    ):
        (
            ctx.n_iter,
            ctx.grad_p_buffer,
            ctx.grad_q_buffer,
            ctx.device,
            ctx.send_rank,
            ctx.pg,
        ) = (n_iter, grad_p_buffer, grad_q_buffer, device, recv_rank, pg)
        p_buffer[0] = p_buffer[0].to(device)
        q_buffer[0] = q_buffer[0].to(device)
        dist.recv(p_buffer[0], recv_rank, group=pg)
        dist.recv(q_buffer[0], recv_rank, group=pg)
        p_buffer[0] = p_buffer[0].to("cpu")
        q_buffer[0] = q_buffer[0].to("cpu")
        input = PowerSVDDecompress(p_buffer[0], q_buffer[0], input.shape)
        return input

    @staticmethod
    def backward(ctx, grad_backward):
        n_iter, p_buffer, q_buffer, device, send_rank, pg = (
            ctx.n_iter,
            ctx.grad_p_buffer,
            ctx.grad_q_buffer,
            ctx.device,
            ctx.send_rank,
            ctx.pg,
        )
        p, q = PowerSVD(grad_backward, q_buffer, p_buffer, n_iter)
        p = p.to(device)
        q = q.to(device)
        dist.isend(p, send_rank, group=pg)
        dist.isend(q, send_rank, group=pg)
        return grad_backward, None, None, None, None, None, None, None, None


class PowerSVDClientSendLayer(nn.Module):
    def __init__(self, rank, shape, iter, device, send_rank, pg=None) -> None:
        super(PowerSVDClientSendLayer, self).__init__()
        self.p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        self.grad_p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.grad_q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        # print(self.p_buffer.shape,self.q_buffer.shape)
        self.iter = iter
        self.send_rank = send_rank
        self.pg = pg
        self.device = device

    def forward(self, input):
        return PowerSVDSendClient.apply(
            input,
            [self.p_buffer],
            [self.q_buffer],
            self.iter,
            [self.grad_p_buffer],
            [self.grad_q_buffer],
            self.device,
            self.send_rank,
            self.pg,
        )


class PowerSVDClientRecvLayer(nn.Module):
    def __init__(self, rank, shape, iter, device, send_rank, pg=None) -> None:
        super(PowerSVDClientRecvLayer, self).__init__()
        self.p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        self.grad_p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.grad_q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        # print(self.p_buffer.shape,self.q_buffer.shape)
        self.iter = iter
        self.send_rank = send_rank
        self.pg = pg
        self.device = device

    def forward(self, input):
        return PowerSVDRecvClient.apply(
            input,
            [self.p_buffer],
            [self.q_buffer],
            self.iter,
            [self.grad_p_buffer],
            [self.grad_q_buffer],
            self.device,
            self.send_rank,
            self.pg,
        )


class PowerSVDRecvServer(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        p_buffer,
        q_buffer,
        n_iter,
        grad_p_buffer,
        grad_q_buffer,
        recv_rank,
        pg=None,
    ):
        ctx.p_buffer, ctx.q_buffer = grad_p_buffer, grad_q_buffer
        ctx.n_iter, ctx.send_rank, ctx.pg = n_iter, recv_rank, pg
        dist.recv(p_buffer[0], recv_rank, group=pg)
        dist.recv(q_buffer[0], recv_rank, group=pg)
        input = PowerSVDDecompress(p_buffer[0], q_buffer[0], input.shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        p_buffer, q_buffer = ctx.p_buffer, ctx.q_buffer
        n_iter, send_rank, pg = ctx.n_iter, ctx.send_rank, ctx.pg
        p, q = PowerSVD(grad_output, q_buffer, p_buffer, n_iter)
        dist.isend(p, send_rank, group=pg)
        dist.isend(q, send_rank, group=pg)
        return grad_output, None, None, None, None, None, None, None


class PowerSVDSendServer(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        p_buffer,
        q_buffer,
        n_iter,
        grad_p_buffer,
        grad_q_buffer,
        send_rank,
        pg=None,
    ):
        ctx.grad_p_buffer, ctx.grad_q_buffer = grad_p_buffer, grad_q_buffer
        ctx.recv_rank, ctx.pg = send_rank, pg
        p, q = PowerSVD(input, q_buffer, p_buffer, n_iter)
        dist.isend(p, send_rank, group=pg)
        dist.isend(q, send_rank, group=pg)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        p_buffer, q_buffer = ctx.grad_p_buffer, ctx.grad_q_buffer
        recv_rank, pg = ctx.recv_rank, ctx.pg
        dist.recv(p_buffer[0], recv_rank, group=pg)
        dist.recv(q_buffer[0], recv_rank, group=pg)
        grad_output = PowerSVDDecompress(p_buffer[0], q_buffer[0], grad_output.shape)
        return grad_output, None, None, None, None, None, None, None


class PowerSVDServerSendLayer(nn.Module):
    def __init__(self, rank, shape, iter, send_rank, pg=None) -> None:
        super(PowerSVDServerSendLayer, self).__init__()
        self.p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        self.grad_p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.grad_q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        # print(self.p_buffer.shape,self.q_buffer.shape)
        self.iter = iter
        self.send_rank = send_rank
        self.pg = pg

    def forward(self, input):
        return PowerSVDSendServer.apply(
            input,
            [self.p_buffer],
            [self.q_buffer],
            self.iter,
            [self.grad_p_buffer],
            [self.grad_q_buffer],
            self.send_rank,
            self.pg,
        )


class PowerSVDServerRecvLayer(nn.Module):
    def __init__(self, rank, shape, iter, recv_rank, pg=None) -> None:
        super(PowerSVDServerRecvLayer, self).__init__()
        self.p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        self.grad_p_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[2] * shape[3]), rank))
        )
        self.grad_q_buffer = torch.nn.Parameter(
            torch.randn((int(shape[0]), int(shape[1]), rank))
        )
        # print(self.p_buffer.shape,self.q_buffer.shape)
        self.iter = iter
        self.recv_rank = recv_rank
        self.pg = pg

    def forward(self, input):
        return PowerSVDRecvServer.apply(
            input,
            [self.p_buffer],
            [self.q_buffer],
            self.iter,
            [self.grad_p_buffer],
            [self.grad_q_buffer],
            self.recv_rank,
            self.pg,
        )
