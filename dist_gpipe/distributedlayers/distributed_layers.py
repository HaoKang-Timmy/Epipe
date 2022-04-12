from torch import autograd
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        # out = F.relu(x)
        # out = F.avg_pool2d(out, 4)
        out = x.view(x.size(0), -1)
        return out


class ForwardSend_BackwardReceive(autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, from_rank: int, to_rank: int, self_rank: int):
        ctx.from_rank, ctx.to_rank, ctx.self_rank = from_rank, to_rank, self_rank
        dist.isend(input, to_rank)
        # print("forward send",input.shape,"from",self_rank,"to",to_rank)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        from_rank, to_rank, self_rank = ctx.from_rank, ctx.to_rank, ctx.self_rank
        # output = torch.empty(size).cuda(int(self_rank))
        dist.recv(grad_output, int(from_rank))
        # print("backward recv",output.shape,"from",from_rank,"to",self_rank)
        return grad_output, None, None, None


class ForwardSendLayers(nn.Module):
    def __init__(
        self, size: tuple, from_rank: int, to_rank: int, self_rank: int
    ) -> None:
        super(ForwardSendLayers, self).__init__()
        self.size = size
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.self_rank = self_rank

    def forward(self, input):
        return ForwardSend_BackwardReceive.apply(
            input, self.from_rank, self.to_rank, self.self_rank
        )


class ForwardReceive_BackwardSend(autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, from_rank: int, to_rank: int, self_rank: int):
        # ctx.save_for_backward(
        #     torch.tensor(from_rank), torch.tensor(to_rank), torch.tensor(self_rank)
        # )
        ctx.to_rank = to_rank
        dist.recv(input, from_rank)
        # print("forward recv",input.shape,"from",from_rank,"to",self_rank)
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        to_rank = ctx.to_rank

        dist.isend(grad_output, int(to_rank))
        # print("backward send",grad_output.shape,"to",to_rank)
        return grad_output, None, None, None


class ForwardReceiveLayers(nn.Module):
    def __init__(self, from_rank: int, to_rank: int, self_rank: int) -> None:
        super(ForwardReceiveLayers, self).__init__()
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.self_rank = self_rank

    def forward(self, input):
        return ForwardReceive_BackwardSend.apply(
            input, self.from_rank, self.to_rank, self.self_rank
        )


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Topk(nn.Module):
    def __init__(self, compress_ratio):
        super(Topk, self).__init__()
        self.ratio = compress_ratio

    def forward(self, x):
        shape = x.shape
        x = x.view(-1)
        src, index = torch.topk(x, int(self.ratio * x.shape[0]))
        x = torch.zeros(x.shape).to(x.get_device())
        x = x.scatter_(dim=0, index=index, src=src)
        self.mask = torch.zeros(x.shape).to(x.get_device())
        self.mask.index_fill_(0, index, 1.0)
        self.mask = self.mask.view(shape)
        x = x.view(shape)
        return x

    def backward(self, grad_output):
        return grad_output * self.mask


class TopkFunction(autograd.Function):
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
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.mask, None


class TopkAbs(nn.Module):
    def __init__(self, compress_ratio):
        super(TopkAbs, self).__init__()
        self.ratio = compress_ratio

    def forward(self, x):
        return TopkFunction.apply(x, self.ratio)


class QuantFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits):
        max, min = input.max(), input.min()
        step = torch.tensor([(max - min) / (pow(2, bits) - 1)]).to(input.get_device())
        min = torch.tensor([min.item()]).to(input.get_device())
        shape = torch.tensor(input.shape).to(input.get_device())
        output = torch.cat(
            [torch.round((input.view(-1) - min) / step), step, min, shape], 0
        )
        # print("quant forward")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, shape, min, step = (
            grad_output[:-6],
            grad_output[-4:],
            grad_output[-5],
            grad_output[-6],
        )
        # output = (output + min) * step
        output = output * step + min
        output = output.view(tuple(shape.int()))
        # print("quant backward")
        return output, None


class DeQuantFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits):
        ctx.bits = bits
        dequant_input, shape, min, step = input[:-6], input[-4:], input[-5], input[-6]
        # dequant_input = (dequant_input + min) * step
        dequant_input = dequant_input * step + min
        dequant_input = dequant_input.view(tuple(shape.int()))
        return dequant_input

    @staticmethod
    def backward(ctx, grad_output):
        max, min = grad_output.max(), grad_output.min()
        step = torch.tensor([(max - min) / (pow(2, ctx.bits) - 1)]).to(
            grad_output.get_device()
        )
        min = torch.tensor([min.item()]).to(grad_output.get_device())
        shape = torch.tensor(grad_output.shape).to(grad_output.get_device())
        output = torch.cat(
            [torch.round((grad_output.view(-1) - min) / step), step, min, shape], 0
        )
        return output, None


class QuantLayer(nn.Module):
    def __init__(self, bits):
        super(QuantLayer, self).__init__()
        self.bits = bits

    def forward(self, input):
        # if self.training:
        return QuantFunction.apply(input, self.bits)

    # else:
    #     print("evaluate")
    #     return input


class DeQuantLayer(nn.Module):
    def __init__(self, bits) -> None:
        super(DeQuantLayer, self).__init__()
        self.bits = bits

    def forward(self, input):
        # if self.training:
        return DeQuantFunction.apply(input, self.bits)
