from os import stat
import torch.nn.functional as F
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist


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
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.mask, None


class TopkLayer(nn.Module):
    def __init__(self, compress_ratio):
        super(TopkLayer, self).__init__()
        self.ratio = compress_ratio

    def forward(self, x):
        return TopkPruning.apply(x, self.ratio)


# class QuantFunction(autograd.Function):
#     @staticmethod
#     def forward(ctx,input,bits):
#         max, min = input.max(), input.min()
#         step = torch.tensor([(max - min) / (pow(2,bits)-1)]).to(input.get_device())
#         min = torch.tensor([min.item()]).to(input.get_device())
#         shape = torch.tensor(input.shape).to(input.get_device())
#         output = torch.cat([torch.round((input.view(-1)-min)/step),step,min,shape],0)
#         # print("quant forward")
#         return output
#     @staticmethod
#     def backward(ctx,grad_output):
#         output,shape,min,step = grad_output[:-6],grad_output[-4:],grad_output[-5],grad_output[-6]
#         # output = (output + min) * step
#         output = output * step + min
#         output = output.view(tuple(shape.int()))
#         # print("quant backward")
#         return output,None
class Quantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, min, step, bits):
        ctx.bits, ctx.min, ctx.step = bits, min, step
        if bits <= 8:
            output = torch.round((input - min) / step - pow(2, bits - 1)).type(
                torch.cuda.CharTensor
            )
        else:
            output = torch.round((input - min) / step - pow(2, bits - 1)).type(
                torch.cuda.HalfTensor
            )  # 16
        pass

    @staticmethod
    def backward(ctx, grad_output):
        bits, min, step = ctx.bits, ctx.min, ctx.step
        output = (
            grad_output.type(torch.cuda.FloatTensor) + pow(2, bits - 1)
        ) * step + min
        return output, None, None, None, None


class QuantizationLayer(nn.Module):
    def __init__(self, bits, momentum=0.1):
        super(QuantizationLayer, self).__init__()
        self.running_min = 0.0
        self.time = 0
        self.running_step = 0.0
        self.momentum = momentum
        self.bits = bits

    def forward(self, x):
        min, max = x.min(), x.max()
        step = torch.tensor((max - min) / (pow(2, self.bits) - 1))
        # min = torch.tensor([min.item()]).to(x.get_device())
        if self.training:
            self.time = self.time + 1
            if self.time == 1:
                self.running_min, self.running_step = (
                    min.item(),
                    step.item(),
                )  # need to change to CPU?
            else:
                self.running_min = (
                    self.running_min * (1 - self.momentum) + min.item() * self.momentum
                )
                self.running_step = (
                    self.running_step * (1 - self.momentum)
                    + step.item() * self.momentum
                )
            return Quantization.apply(x, min, step, self.bits)
        else:
            self.time = 0
            return Quantization.apply(x, self.running_min, self.running_step, self.bits)


class RemoteQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, min, step, bits, from_rank, to_rank):
        ctx.from_rank, ctx.bits = from_rank, bits
        # if bits <= 8:
        #     output = torch.round((input - min) / step - pow(2, bits - 1)).type(
        #         torch.cuda.CharTensor
        #     )
        # else:
        output = torch.round((input - min) / step - pow(2, bits - 1)).type(
            torch.cuda.HalfTensor
        )  # 16
        
        dist.isend(torch.tensor(min.item()).to(input.get_device()), to_rank)
        dist.isend(torch.tensor(step.item()).to(input.get_device()), to_rank)
        dist.isend(output, to_rank)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        from_rank, bits = ctx.from_rank, ctx.bits
        min = torch.tensor(0.0).to(grad_output.get_device())
        step = torch.tensor(0.0).to(grad_output.get_device())
        # if bits <= 8:
        #     grad_output = grad_output.type(torch.cuda.CharTensor)
        # else:
        grad_output = grad_output.type(torch.cuda.HalfTensor)
        dist.recv(min, from_rank)
        dist.recv(step, from_rank)
        dist.recv(grad_output, from_rank)
        output = (
            grad_output.type(torch.cuda.FloatTensor) + pow(2, bits - 1)
        ) * step + min
        return output, None, None, None, None, None


class RemoteQuantizationLayer(nn.Module):
    def __init__(self, bits, from_rank: int, to_rank: int, momentum=0.1):
        super(RemoteQuantizationLayer, self).__init__()
        self.running_min = torch.tensor(0.0)
        self.time = 0
        self.running_step = torch.tensor(0.0)
        self.momentum = momentum
        self.bits = bits
        self.from_rank = from_rank
        self.to_rank = to_rank

    def forward(self, x):
        if self.training:
            min, max = x.min(), x.max()
            step = (max - min) / (pow(2, self.bits) - 1)
            self.time = self.time + 1
            if self.time == 1:
                self.running_min, self.running_step = (
                    min,
                    step,
                )  # need to change to CPU?
            else:
                self.running_min = (
                    self.running_min * (1 - self.momentum) + min.item() * self.momentum
                )
                self.running_step = (
                    self.running_step * (1 - self.momentum)
                    + step.item() * self.momentum
                )
            output = RemoteQuantization.apply(x, min, step, self.bits,self.from_rank,self.to_rank)
            return output
        else:
            # self.time = 0

            output = RemoteQuantization.apply(
                x, self.running_min, self.running_step, self.bits,self.from_rank,self.to_rank
            )
            return output




##TODO
class RemoteDeQuantization(autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input,
        bits,
        running_min,
        running_step,
        momentum,
        from_rank: int,
        to_rank: int,
        train=None,
        init=None,
    ):
        # ctx.bits,ctx.to_rank = bits,to_rank
        # if init is not None:
        #     ctx.running_min = 0.0
        #     ctx.running_step = 0.0
        (
            ctx.train,
            ctx.to_rank,
            ctx.init,
            ctx.bits,
            ctx.running_min,
            ctx.running_step,
            ctx.momentum,
        ) = (train, to_rank, init, bits, running_min, running_step, momentum)
        min = torch.tensor(0.0).to(input.get_device())
        step = torch.tensor(0.0).to(input.get_device())
        # if bits <= 8:
        #     input = input.type(torch.cuda.CharTensor)
        # else:
        input = input.type(torch.cuda.HalfTensor)
        dist.recv(min, from_rank)
        dist.recv(step, from_rank)
        dist.recv(input, from_rank)
        input = input.type(torch.cuda.FloatTensor)
        # print(input,"recv from")
        # TODO recv datatype change?
        output = (input + pow(2, bits - 1)) * step + min
        return output

    @staticmethod
    def backward(ctx, grad_output):
        train, to_rank, init, bits, running_min, running_step, momentum = (
            ctx.train,
            ctx.to_rank,
            ctx.init,
            ctx.bits,
            ctx.running_min,
            ctx.running_step,
            ctx.momentum,
        )
        if train is not None:
            min, max = grad_output.min(), grad_output.max()
            step = (max - min) / (pow(2, bits) - 1)
            if init is not None:

                ctx.running_min[0], ctx.running_step[0] = min, step
            else:
                ctx.running_min[0], ctx.running_step[0] = (
                    ctx.running_min[0] * (1 - momentum) + min * momentum,
                    ctx.running_step[0] * (1 - momentum) + step * momentum,
                )
        else:
            min = running_min
            step = running_step
        # if bits <= 8:
        #     output = torch.round((grad_output - min) / step - pow(2, bits - 1)).type(
        #         torch.cuda.CharTensor
        #     )
        # else:
        output = torch.round((grad_output - min) / step - pow(2, bits - 1)).type(
            torch.cuda.HalfTensor
        )
        dist.isend(min, to_rank)
        dist.isend(step, to_rank)
        dist.isend(output, to_rank)
        return output,None,None,None,None,None,None,None,None


class RemoteDeQuantizationLayer(nn.Module):
    def __init__(self, bits, from_rank: int, to_rank: int, momentum=0.1):
        super(RemoteDeQuantizationLayer, self).__init__()
        self.bits = bits
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.momentum = momentum
        self.running_min = torch.tensor([0.0])
        self.running_step = torch.tensor([0.0])
        self.time = 0

    def forward(self, input):
        if self.training:
            self.time = self.time + 1
            if self.time == 1:

                return RemoteDeQuantization.apply(
                    input,
                    self.bits,
                    self.running_min,
                    self.running_step,
                    self.momentum,
                    self.from_rank,
                    self.to_rank,
                    1,
                    1,
                )
            else:
                return RemoteDeQuantization.apply(
                    input,
                    self.bits,
                    self.running_min,
                    self.running_step,
                    self.momentum,
                    self.from_rank,
                    self.to_rank,
                    1,
                )
        else:
            # self.time = 0 runing variable changes every time
            return RemoteDeQuantization.apply(
                input,
                self.bits,
                self.running_min,
                self.running_step,
                self.momentum,
                self.from_rank,
                self.to_rank,
            )

        # step
        # output = (grad_output.type(torch.cuda.FloatTensor) + pow(2,bits-1)) * step +min
        # return output, None, None, None, None


# class Test(autograd.Function):
#     @staticmethod
#     def forward(ctx, input, running_min, running_step):
#         ctx.running_min = running_min
#         ctx.running_step = running_step
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         ctx.running_min[0] = ctx.running_min[0] + 1
#         ctx.running_step[0] = ctx.running_step[0] + 1
#         return grad_output, None, None


# class TestLayer(nn.Module):
#     def __init__(self) -> None:
#         super(TestLayer, self).__init__()
#         self.register_buffer("running_min", torch.tensor([0.0]))
#         self.register_buffer("running_step", torch.tensor([0.0]))

#     def forward(self, input):
#         return Test.apply(input, self.running_min, self.running_step)


# test_layer = TestLayer()

# x = torch.rand([2, 2]).requires_grad_()
# y = torch.ones([2, 2]).requires_grad_()
# z = x + y

# z = test_layer(z)
# z = z.sum()
# z.backward()
# z.backward()
# print(test_layer.running_min)
# x = test.apply(x)
# x.backward()
# x.backward()
# x.backward()

