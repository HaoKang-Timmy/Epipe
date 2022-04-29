import torch.nn.functional as F
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist
import time


def q10toint32():
    pass


def q12toint32():
    pass


def QuantizationGPU(input, bits):

    min, max = input.min(), input.max()
    step = (max - min) / (pow(2, bits) - 1)
    output = torch.round((input - min) / step - pow(2, bits - 1))
    # print("quant 16bits",output)
    if bits <= 8:
        output = output.type(torch.int8)
    # elif bits == 10:
    #     output = q10toint32()
    # elif bits == 12:
    #     output = q12toint32()#TODO
    elif bits <= 16:
        output = output.type(torch.int16)
        output = output.view(dtype=torch.int8)
    min = min.to(input.get_device())
    step = step.to(input.get_device())
    return output, min, step


def QtensorSendonGPU(input, min, step, send_rank, pg=None):

    dist.isend(min, send_rank, group=pg)
    dist.isend(step, send_rank, group=pg)
    dist.isend(input, send_rank, group=pg)


def QtensorRecvonGPU1(input, bits, min, step, recv_rank, pg=None):
    # print(input.shape)
    # if bits <= 8:
    #     input =input.type(torch.int8)
    # # elif bits == 10:
    # #     input = int32to10()TODO
    # # elif bits == 12:
    # #     input = int32to12()
    # elif bits <= 16:
    #     input =input.type(torch.int16)
    #     input = input.view(dtype = torch.int8)
    # print(pg)
    dist.recv(min, recv_rank, group=pg)
    dist.recv(step, recv_rank, group=pg)
    dist.recv(input, recv_rank, group=pg)
    return min, step, input


def DequantizationonGPU(input, bits, min, step):
    # print("recv 16",input)
    output = (
        (input.type(torch.float32) + pow(2, bits - 1)) * step + min
    ).requires_grad_()
    return output


def SortQuantization(input, bits, split_bits, min_step):
    shape = input.shape
    input = input.view(-1)

    src, index = torch.sort(input, dim=0)

    index = torch.tensor_split(index, 2**split_bits)
    src = torch.tensor_split(src, 2**split_bits)
    if bits + split_bits <= 8:
        output = input.type(torch.int8)
    else:
        output = input.type(torch.int16)
    output = output.view(-1)

    for i in range(2**split_bits):

        if bits + split_bits == 8 or bits + split_bits == 16:

            min, max = src[i].min(), src[i].max()

            if min != max:

                step = (max - min) / (pow(2, bits) - 1)

                min_step[i, 0], min_step[i, 1] = min, step

                temp_src = torch.round((src[i] - min) / step) - pow(
                    2, bits + split_bits - 1
                )
                temp_src += pow(2, bits) * i

            else:
                some = time.time()
                min_step[i, 0], min_step[i, 1] = min, 0.0
                temp_src = src[i] - src[i] - pow(2, bits + split_bits - 1)
                temp_src += pow(2, bits) * i
                print(time.time() - some)

        else:
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)
                min_step[i, 0], min_step[i, 1] = min, step
                temp_src = torch.round((src[i] - min) / step)
                temp_src += pow(2, bits) * i
            else:
                min_step[i, 0], min_step[i, 1] = min, 0.0
                temp_src = 1 + src[i] - src[i]
                temp_src += pow(2, bits) * i
        temp_src = temp_src.type(output.dtype)
        output.scatter_(0, index[i], temp_src)

    output = output.view(shape)
    if bits + split_bits > 8 and bits + split_bits <= 16:
        output = output.view(dtype=torch.int8)

    return min_step, output


def SortDeQuantization(recv, bits, split_bits, min_step, grad_output):
    shape = grad_output.shape
    if bits + split_bits > 8 and bits + split_bits <= 16:
        recv = recv.view(dtype=torch.int16)
    src, index = torch.sort(recv, dim=0)
    index = torch.tensor_split(index, 2**split_bits)
    src = torch.tensor_split(src, 2**split_bits)
    for i in range(2**split_bits):
        temp_src = src[i].type(torch.float32)
        if bits + split_bits == 8 or bits + split_bits == 16:
            if min_step[i, 1] == 0:
                # step = 0
                # temp_src = min_step[i,0]
                temp_src.fill_(min_step[i, 0])
            else:
                offset = pow(2, bits + split_bits - 1) - pow(2, bits) * i
                temp_src += offset
                temp_src *= min_step[i, 1]
                temp_src += min_step[i, 0]
        else:
            if min_step[i, 1] == 0:
                temp_src.fill_(min_step[i, 0])
            else:
                offset = -pow(2, bits) * i
                temp_src += offset
                temp_src *= min_step[i, 1]
                temp_src += min_step[i, 0]
        grad_output.scatter_(0, index[i], temp_src)
    return grad_output


def FastDeQuantization(recv: torch.tensor, bits, split_bits, min_step, grad_output):
    if bits + split_bits > 8 and bits + split_bits <= 16:
        recv = recv.view(dtype=torch.int16)
    recv = recv.type(torch.long)

    for i in range(2**split_bits):

        if bits + split_bits == 8 or bits + split_bits == 16:
            upperbound = -pow(2, bits + split_bits - 1) + pow(2, bits) * (i + 1)
            lowerbound = -pow(2, bits + split_bits - 1) + pow(2, bits) * i
            # start = time.time()
            temp = torch.where((recv < upperbound) & (recv >= lowerbound), recv, 0)
            # upper = recv < upperbound
            # lower = recv >= lowerbound
            # mask = upper * lower
            # temp = torch.masked_select(recv,mask)
            # slower than where, not used
            # print(time.time() - start)
            temp = temp.type(torch.float)
            # print(temp)
            indexs = torch.nonzero(temp)
            indexs = indexs.view(-1)
            # print(indexs)
            temp = torch.index_select(temp, 0, indexs)

            if min_step[i, 1] == 0:
                temp.fill_(min_step[i, 0])
            else:
                offset = pow(2, bits + split_bits - 1) - pow(2, bits) * i
                temp += offset
                temp *= min_step[i, 1]
                temp += min_step[i, 0]
        else:
            upperbound = pow(2, bits) * (i + 1)
            lowerbound = pow(2, bits) * i
            temp = torch.where((recv < upperbound) & (recv >= lowerbound), recv, 0)
            temp = temp.type(torch.float)
            indexs = torch.nonzero(temp)
            indexs = indexs.view(-1)
            temp = torch.index_select(temp, 0, indexs)
            if min_step[i, 1] == 0:
                temp.fill_(min_step[i, 0])
            else:
                offset = -pow(2, bits) * i
                temp += offset
                temp *= min_step[i, 1]
                temp += min_step[i, 0]

        grad_output.scatter_(0, indexs, temp)
    return grad_output
