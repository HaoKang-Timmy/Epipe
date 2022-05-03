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
    # print('sq',input)
    input = input.view(-1)
    src, index = torch.sort(input, dim=0)
    # print(src)
    index = torch.tensor_split(index, 2 ** split_bits)
    src = torch.tensor_split(src, 2 ** split_bits)
    if bits + split_bits <= 8:
        output = input.type(torch.int8)
    else:
        output = input.type(torch.int16)
    output = output.view(-1)

    for i in range(2 ** split_bits):

        if bits + split_bits == 8 or bits + split_bits == 16:

            min, max = src[i].min(), src[i].max()

            # if min != max:

            step = (max - min) / (pow(2, bits) - 1)

            min_step[i, 0], min_step[i, 1] = min, step
            if step != 0.0:
                temp_src = torch.round((src[i] - min) / step)

            else:
                temp_src = src[i] - src[i]
            temp_src += -pow(2, bits + split_bits - 1) + pow(2, bits) * i
        else:
            min, max = src[i].min(), src[i].max()
            # if min != max:
            step = (max - min) / (pow(2, bits) - 1)
            min_step[i, 0], min_step[i, 1] = min, step
            if step != 0.0:
                temp_src = torch.round((src[i] - min) / step)
            else:
                temp_src = src[i] - src[i]
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

    index = torch.tensor_split(index, 2 ** split_bits)
    src = torch.tensor_split(src, 2 ** split_bits)

    for i in range(2 ** split_bits):
        # torch.cuda.synchronize()
        # start = time.time()
        temp_src = src[i].type(torch.float32)

        if bits + split_bits == 8 or bits + split_bits == 16:

            # if min_step[i, 1] == 0:
            #     # step = 0
            #     # temp_src = min_step[i,0]
            #     temp_src.fill_(min_step[i, 0])
            # else:
            # torch.cuda.synchronize()
            # print(time.time()- start)
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
    # print("deq",grad_output)

    return grad_output


def FastDequantizationGPU(recv, bits, split_bits, min_step, lower_bound, upper_bound):
    shape = recv.shape
    if bits + split_bits > 8 and bits + split_bits <= 16:
        recv = recv.view(dtype=torch.int16)
    recv = recv.view(-1)
    src, index = torch.sort(recv, dim=0)
    # print(src,index)
    recv = recv.type(torch.float32)
    # index = torch.tensor_split(index, 2**split_bits)
    # src = torch.tensor_split(src, 2**split_bits)
    for i in range(2 ** split_bits):

        # lowerbound, upperbound = min_step[i, 2].type(torch.int).cpu(), min_step[i,3].type(torch.int).cpu()

        # split_src = src[i].type(torch.float32)

        # print(lowerbound,upperbound)
        split_index = index[lower_bound[i] : upper_bound[i]]

        # print(upperbound - lowerbound)
        split_src = src[lower_bound[i] : upper_bound[i]]
        split_src = split_src.type(torch.float32)
        # print(split_src)
        # print(lower_bound,upper_bound)
        if bits + split_bits == 8 or bits + split_bits == 16:

            offset = pow(2, bits + split_bits - 1) - pow(2, bits) * i
            split_src += offset
            # print(split_src)
            split_src *= min_step[i, 1]
            split_src += min_step[i, 0]
        else:
            # if min_step[i, 1] == 0:
            #     temp_src.fill_(min_step[i, 0])
            # else:
            offset = -pow(2, bits) * i
            split_src += offset
            split_src *= min_step[i, 1]
            split_src += min_step[i, 0]
        recv.scatter_(0, split_index, split_src)
    # print("deq",grad_output)
    recv = recv.view(shape)
    return recv


def SortDequantizationCPU(recv: torch.tensor, bits, split_bits, min_step, grad_output):
    shape = recv.shape
    recv = recv.view(-1)
    grad_output = grad_output.view(-1)
    if bits + split_bits > 8 and bits + split_bits <= 16:
        recv = recv.view(dtype=torch.int16)
    recv = recv.type(torch.long)

    for i in range(2 ** split_bits):
        if bits + split_bits == 8 or bits + split_bits == 16:
            upperbound = -pow(2, bits + split_bits - 1) + pow(2, bits) * (i + 1)
            lowerbound = -pow(2, bits + split_bits - 1) + pow(2, bits) * i
            # print(upperbound,lowerbound)
            temp = torch.where(
                (recv < upperbound) & (recv >= lowerbound), recv, -100000
            )
            indexs = (temp != -100000).nonzero()
            indexs = indexs.view(-1)
            temp = torch.index_select(temp, 0, indexs)
            offset = pow(2, bits + split_bits - 1) - pow(2, bits) * i
            temp += offset
            # print(min_step[i, 1],min_step[i, 0])
            temp = temp.type(torch.float)
            temp *= min_step[i, 1]
            temp += min_step[i, 0]
        else:
            upperbound = pow(2, bits) * (i + 1)
            lowerbound = pow(2, bits) * i
            temp = torch.where((recv < upperbound) & (recv >= lowerbound), recv, 0)

            indexs = torch.nonzero(temp)
            indexs = indexs.view(-1)
            temp = torch.index_select(temp, 0, indexs)
            offset = -pow(2, bits) * i
            temp += offset
            temp = temp.type(torch.float)
            temp *= min_step[i, 1]
            temp += min_step[i, 0]

        grad_output.scatter_(0, indexs, temp)

    grad_output = grad_output.view(shape)
    recv = recv.view(shape)
    # print("fastdeq",grad_output)
    return grad_output


def FastQuantizationCPU(input, bits, split_bits, min_step):
    # print("fastq",input)
    lowerbound = 0.0
    upperbound = 0.0
    shape_tensor = input.shape
    batch = shape_tensor[0]
    input = input.view(-1).type(torch.double)
    separate = torch.tensor(-1e9).type(torch.double)
    shape = int(input.shape[0])
    if bits + split_bits <= 8:
        output = input.type(torch.int8)
    else:
        output = input.type(torch.int16)
    for i in range(2 ** split_bits):

        if i == 2 ** split_bits - 1:
            kthvalue = input.max()
        else:
            input = input.view(batch, -1)
            kthvalue, indice = torch.kthvalue(
                input[0],
                int(shape * (i + 1) / (2 ** split_bits) / batch),
                keepdim=False,
            )

            input = input.view(-1)

            kthvalue = kthvalue.type(torch.double)
        if kthvalue == separate:
            temp = torch.where((input == kthvalue), input, -1000000.0)
        else:
            temp = torch.where(
                (input <= kthvalue) & (input > separate), input, -1000000.0
            )
        temp = temp.type(torch.float)
        # print(temp)
        indexs = (temp != -1000000.0).nonzero()
        indexs = indexs.view(-1)
        temp = torch.index_select(temp, 0, indexs)
        if bits + split_bits == 8 or bits + split_bits == 16:
            offset = -pow(2, bits + split_bits - 1) + pow(2, bits) * i
            min = temp.min()
            step = (kthvalue - min) / (pow(2, bits) - 1)
            min_step[i, 0], min_step[i, 1] = min, step
            if step != 0.0:
                temp = torch.round((temp - temp.min()) / step)
            else:
                temp = temp - temp
            temp += offset
        else:
            offset = pow(2, bits) * i
            min = temp.min()
            step = (kthvalue - min) / (pow(2, bits) - 1)
            min_step[i, 0], min_step[i, 1] = min, step
            if step != 0.0:
                temp = torch.round((temp - min) / step)
            else:
                temp = temp - temp
            temp += pow(2, bits) * i

        # print(temp)
        temp = temp.type(output.dtype)
        output.scatter_(0, indexs, temp)
        # print(time.time() - start)
        # print(kthvalue, separate)
        if kthvalue != separate:
            lowerbound = upperbound
            upperbound += temp.shape[0]
            # print(upperbound,lowerbound)
        min_step[i, 2], min_step[i, 3] = lowerbound, upperbound
        separate = kthvalue

    output = output.view(shape_tensor)
    if bits + split_bits > 8 and bits + split_bits <= 16:
        output = output.view(dtype=torch.int8)
    # print(min_step)
    # print(output)
    return min_step, output


def ParallelSortQuantization(input, bits, split_bits, min_step):
    shape_input = input.shape
    input = input.view(2 ** split_bits, -1).type(torch.double)
