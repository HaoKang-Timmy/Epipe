from os import stat
import torch.nn.functional as F
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist
def create_sparse(input:torch.tensor,bit_saving = True):
    shape = input.shape
    input = input.view(-1)
    index = input.nonzero()
    index = index.view(-1)
    if bit_saving is True:
        index = index.type(torch.bfloat16)
    src = input.index_select(0,index)
    input = input.view(shape)
    return shape,index,src
def unzip_sparse(input,index,src,shape):
    input = input.view(-1)
    input.scatter_(0,index,src)
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
## These are gloo api
def QuantizationonGPU(input:torch.tensor,bits,min_step):
    
    min,max = input.min(), input.max()
    step = (max - min) / (pow(2, bits) - 1)
    output = torch.round((input - min) / step - pow(2, bits - 1))
    if bits <= 8:
        output = output.type(torch.cuda.CharTensor)
    elif bits <= 16:
        output = output.type(torch.cuda.ShortTensor)
    else:
        output = output.type(torch.cuda.IntTensor)
    
    min_step[0] = min.item()
    min_step[1] = step.item()
    return min_step,output
def DequantizationonGPU(input:torch.tensor,bits,min,step):
    input = input.type(torch.cuda.FloatTensor)

    output = (input + pow(2, bits - 1)) * step + min
    output = output.requires_grad_()
    return output
class QSend(autograd.Function):
    @staticmethod
    def forward(ctx,input,bits,min_step,send_rank,rank):
        ctx.bits,ctx.recv_rank,ctx.rank,ctx.min_step = bits, send_rank, rank, min_step
        min_step, output = QuantizationonGPU(input,bits,min_step)
        dist.send(min_step.cpu(),send_rank)
        dist.send(output.cpu(),send_rank)
        return input
    @staticmethod
    def backward(ctx,grad_output):
        bits, send_rank, rank, min_step = ctx.bits,ctx.recv_rank,ctx.rank,ctx.min_step
        input = grad_output.cpu()
        if bits <= 8:
            input = input.type(torch.int8)
        elif bits <= 16:
            input = input.type(torch.int16)
        else:
            input = grad_output.type(torch.int32)
        min_step_cpu = min_step.cpu()
        dist.recv(min_step_cpu,send_rank)
        dist.recv(input,send_rank)
        input = input.to(rank)
        min_step = min_step_cpu.to(rank)
        grad_output = DequantizationonGPU(input, bits,min_step[0],min_step[1])
        # print(grad_output)
        return grad_output,None, None, None, None
class QSendSparse(autograd.Function):
    @staticmethod
    def forward(ctx,input,bits,min_step,send_rank,rank):
        ctx.bits,ctx.recv_rank,ctx.rank,ctx.min_step = bits, send_rank, rank, min_step
        min_step, output = QuantizationonGPU(input,bits,min_step)
        shape, index, src = create_sparse(output)
        dist.send(min_step.cpu(),send_rank)
        index = index.cpu()
        src = src.cpu()
        dist.send(index,send_rank)
        dist.send(src,send_rank)
        ctx.index,ctx.src = index,src
        return input
    @staticmethod
    def backward(ctx,grad_output):
        bits, recv_rank, rank, min_step = ctx.bits,ctx.recv_rank,ctx.rank,ctx.min_step
        index, src = ctx.index, ctx.src
        min_step = min_step.cpu()
        dist.recv(min_step,recv_rank)
        dist.recv(index,recv_rank)
        dist.recv(src,recv_rank)
        input = grad_output.cpu()
        if bits <= 8:
            input = input.type(torch.int8)
        elif bits <= 16:
            input = input.type(torch.int16)
        else:
            input = grad_output.type(torch.int32)
        input = unzip_sparse(input,index,src,input.shape)
        input = input.to(rank)
        min_step = min_step.to(rank)
        grad_output = DequantizationonGPU(input, bits,min_step[0],min_step[1])
        return grad_output,None, None, None, None
        #TODO not finish yet



    

class QSendLayer(nn.Module):
    def __init__(self, bits,send_rank,rank,sparse = False) -> None:
        super(QSendLayer, self).__init__()
        self.bits = bits
        self.min_step = torch.tensor([0.,0.])
        self.send_rank = send_rank
        self.rank = rank
        self.sparse = sparse
    def forward(self,input):
        if self.sparse is False:
            return QSend.apply(input,self.bits,self.min_step,self.send_rank,self.rank)
        else:
            return QSendSparse.apply(input,self.bits,self.min_step,self.send_rank,self.rank)

class Qrecv(autograd.Function):
    @staticmethod
    def forward(ctx,input,bits,min_step,recv_rank,rank):
        ctx.bits,ctx.send_rank,ctx.rank,ctx.min_step = bits, recv_rank, rank, min_step
        min_step_cpu = min_step.cpu()
        recv = input.cpu()
        if bits <= 8:
            recv = recv.type(torch.int8)
        elif bits <= 16:
            recv = recv.type(torch.int16)
        else:
            recv = recv.type(torch.int32)
        
        dist.recv(min_step_cpu,recv_rank)
        dist.recv(recv,recv_rank)
        min_step = min_step_cpu.to(rank)
        recv = recv.to(rank)
        input = DequantizationonGPU(recv,bits,min_step[0],min_step[1])
        return input
    @staticmethod
    def backward(ctx,grad_output):
        bits, recv_rank, rank, min_step = ctx.bits,ctx.send_rank,ctx.rank,ctx.min_step
        min_step ,output = QuantizationonGPU(grad_output,bits,min_step)
        dist.send(min_step.cpu(),recv_rank)
        dist.send(output.cpu(),recv_rank)
        return grad_output, None, None, None, None

class QrecvSparse(autograd.Function):
    @staticmethod
    def forward(ctx,input,bits,min_step,recv_rank,rank):
        ctx.bits,ctx.send_rank,ctx.rank,ctx.min_step = bits, recv_rank, rank, min_step
        min_step_cpu = min_step.cpu()
        recv = input.cpu()
        if bits <= 8:
            recv = recv.type(torch.int8)
        elif bits <= 16:
            recv = recv.type(torch.int16)
        else:
            recv = recv.type(torch.int32)
            #TODO how to know the size of src and index
        


class QrecvLayer(nn.Module):
    def __init__(self,bits,recv_rank,rank) -> None:
        super(QrecvLayer,self).__init__()
        self.bits = bits
        self.recv_rank = recv_rank
        self.rank = rank
        self.min_step = torch.tensor([0.,0.])
    def forward(self,input):
        return Qrecv.apply(input,self.bits,self.min_step,self.recv_rank,self.rank)






















# return cpu tensors
class Quantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, min, step, bits):
        ctx.bits, ctx.min, ctx.step = bits, min, step
        # if bits <= 8:
        #     output = torch.round((input - min) / step - pow(2, bits - 1)).type(
        #         torch.cuda.CharTensor
        #     )
        # else:
        
        output = torch.round((input - min) / step - pow(2, bits - 1)).type(
            torch.cuda.HalfTensor
            )  # 16
        # pass
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bits, min, step = ctx.bits, ctx.min, ctx.step
        output = (
            grad_output.type(torch.cuda.FloatTensor) + pow(2, bits - 1)
        ) * step + min
        return output, None, None, None, None


class QuantizationLayer(nn.Module):
    def __init__(self, bits, momentum=0.1, dynamic = 0):
        super(QuantizationLayer, self).__init__()
        self.running_min = 0.0
        self.time = 0
        self.running_step = 0.0
        self.momentum = momentum
        self.bits = bits
        self.dynamic = dynamic
    def forward(self, x):
        min, max = x.min(), x.max()
        step = torch.tensor((max - min) / (pow(2, self.bits) - 1))
        # min = torch.tensor([min.item()]).to(x.get_device())
        if self.training or self.dynamic == 0:
            self.time = self.time + 1
            if self.time == 1:
                self.running_min, self.running_step = (
                    min.item(),
                    step.item(),
                )  # need to change to CPU?
            elif self.dynamic != 0:
                self.running_min = (
                    self.running_min * (1 - self.momentum) + min.item() * self.momentum
                )
                self.running_step = (
                    self.running_step * (1 - self.momentum)
                    + step.item() * self.momentum
                )
            return Quantization.apply(x, min, step, self.bits),min,step
        else:
            # self.time = 0
            return Quantization.apply(x, self.running_min, self.running_step, self.bits),min,step


class RemoteQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, min, step, bits, from_rank, to_rank):
        ctx.from_rank, ctx.bits = from_rank, bits

        # if bits <= 16:
        #     output1 = torch.round((input - min) / step - pow(2, bits - 1)).type(
        #         torch.cuda.ShortTensor
        #     )  # 16
        # elif bits > 16:
        output1 = torch.round((input - min) / step - pow(2, bits - 1)).type(
            torch.cuda.IntTensor
        )
            # can not send torch.cuda.CharTensor and torch.cuda
        
        
        dist.isend(torch.tensor(min.item()).to(input.get_device()), to_rank)
        dist.isend(torch.tensor(step.item()).to(input.get_device()), to_rank)
        dist.isend(output1, to_rank)

        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        from_rank, bits = ctx.from_rank, ctx.bits
        min = torch.tensor(0.0).to(grad_output.get_device())
        step = torch.tensor(0.0).to(grad_output.get_device())

        # if bits <= 16:
        #     grad_output = grad_output.type(torch.cuda.ShortTensor)
        # else:
        grad_output = grad_output.type(torch.cuda.IntTensor)
        dist.recv(min, from_rank)
        dist.recv(step, from_rank)
        dist.recv(grad_output, from_rank)
        output = (
            grad_output.type(torch.cuda.FloatTensor).requires_grad_() + pow(2, bits - 1)
        ) * step + min
        return output, None, None, None, None, None


class RemoteQuantizationLayer(nn.Module):
    def __init__(self, bits, from_rank: int, to_rank: int, momentum=0.1,dynamic = 0):
        super(RemoteQuantizationLayer, self).__init__()
        self.running_min = torch.tensor(0.0)
        self.time = 0
        self.running_step = torch.tensor(0.0)
        self.momentum = momentum
        self.bits = bits
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.dynamic = dynamic
    def forward(self, x):
        if self.training or self.dynamic == 0:
            min, max = x.min(), x.max()
            step = (max - min) / (pow(2, self.bits) - 1)
            self.time = self.time + 1
            if self.dynamic != 0:
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

class Dequantization(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        bits,
        min,
        max,
        step
    ):
        ctx.bits = bits
        output = (input + pow(2, bits - 1)) * step + min
        return output
    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        min, max = grad_output.min(), grad_output.max()
        step = (max - min) / (pow(2, bits) - 1)
        output1 = torch.round((grad_output - min) / step - pow(2, bits - 1))

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
        # if bits <= 16:
        #     input = input.type(torch.cuda.ShortTensor)
        # else:
        input = input.type(torch.cuda.IntTensor)
        dist.recv(min, from_rank)
        dist.recv(step, from_rank)
        dist.recv(input, from_rank)
        input = input.type(torch.cuda.FloatTensor).requires_grad_()
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
        # if bits <= 16:

        #     output1 = torch.round((grad_output - min) / step - pow(2, bits - 1)).type(
        #         torch.cuda.ShortTensor
        #     )
        # else:
        output1 = torch.round((grad_output - min) / step - pow(2, bits - 1)).type(
            torch.cuda.IntTensor
        )
        dist.isend(min, to_rank)
        dist.isend(step, to_rank)
        dist.isend(output1, to_rank)
        return grad_output * 1.0,None,None,None,None,None,None,None,None


class RemoteDeQuantizationLayer(nn.Module):
    def __init__(self, bits, from_rank: int, to_rank: int, momentum=0.1,dynamic = 0):
        super(RemoteDeQuantizationLayer, self).__init__()
        self.bits = bits
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.momentum = momentum
        self.running_min = torch.tensor([0.0])
        self.running_step = torch.tensor([0.0])
        self.time = 0
        self.dynamic = dynamic

    def forward(self, input):
        if self.training or self.dynamic == 0:
            self.time = self.time + 1
            if self.time == 1 or self.dynamic == 0:

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


class multi_quantization_nosparse(autograd.Function):
    @staticmethod
    # every iter send value and min steps（because this might save transfer bits when sending same value tensors), and finally send keys
    #TODO I changed idea, I want to implement a single one!!! I might do this in the future,could do pipeline as above!!!!!
    def forward(ctx,input,bits,seperation,rank_send):
        ctx.bits, ctx.seperation, ctx.rank_recv = bits, seperation, rank_send
        shape = input.shape
        input = input.view(-1)
        # src, index = torch.topk(torch.abs(input), int(input.shape[0]))
        # src = input.index_select(0,index)
        src, index = torch.topk(input, int(input.shape[0]))
        print("rank",0,"src",src)
        # print(src.size())
        index1 = torch.linspace(start = 0, end = src.size()[0] - 1, steps = 1).type(torch.cuda.LongTensor).to(input.get_device())
        # index = index.chunk(seperation)
        index1 = torch.flip(index1,[0])
        src = src.chunk(seperation)
        index1 = index1.chunk(seperation)
        print("index1",index1)
        min_step = torch.zeros([2,seperation]).to(input.get_device())
        
        for i in range(seperation):
            min, max = src[i].min(), src[i].max()
            if min == max:
                # dist.isend(torch.tensor([min,max]),rank_send)
                min_step[0][i] = min
                min_step[1][i] = min
                temp = src[i]#could be better
                # the receiver will test whether the two are equal, if are, they will put the value to accordingly indexs
            else:
                step = (max - min) / (pow(2, bits) - 1)
                min_step[0][i] = min
                min_step[1][i] = step
                temp = torch.round((src[i] - min) / step) - pow(2, bits - 1)
            src = torch.cat(src,0)
            src.scatter_(0,index1[i],temp)
            src = src.chunk(seperation)
        src = torch.cat(src,0)
        # index = torch.cat(index,0)
        # print("rank",0,"min",min,"step",step)
        print("rank",0,"quant_src",src)
        print("rank",0,"index",index)
        dist.isend(min_step,rank_send)
        dist.isend(index,rank_send)
        dist.isend(src,rank_send)
        return input.view(shape)
    @staticmethod
    def backward(ctx,grad_output):
        bits, seperation, rank_recv = ctx.bits, ctx.seperation, ctx.rank_recv
        grad_shape = grad_output.shape
        grad_output = grad_output.view(-1)
        shape = grad_output.shape
        min_step = torch.zeros([2,seperation]).to(grad_output.get_device())
        index = torch.zeros(shape).type(torch.cuda.LongTensor).to(grad_output.get_device())# need to change type
        src= torch.zeros(shape).to(grad_output.get_device())# need to change type
        # src1 = torch.zeros(shape).to(grad_output.get_device())
        dist.recv(min_step,rank_recv)
        dist.recv(index,rank_recv)
        dist.recv(src, rank_recv)
        # src = src.view(-1)
        # index = index.view(-1)
        src = src.chunk(seperation)
        index = index.chunk(seperation)
        for i in range(seperation):
            min = min_step[0][i]
            step = min_step[1][i]
            if min == step:
                
                src[i].index_fill_(0,index[i],min)
                temp = src[i]
            else:
                temp = (src[i] + pow(2, bits - 1)) * step + min
            src = torch.cat(src,0)
            src.scatter_(0,index[i],temp)
            src = src.chunk(seperation)
        grad_output.scatter_(0,index,src)
        grad_output = grad_output.view(grad_shape)
        return grad_output,None,None,None

class multi_dequantization_nosparse(autograd.Function):
    @staticmethod
    def forward(ctx,input,bits,seperation,rank_recv):
        ctx.bits, ctx.seperation, ctx.rank_send = bits, seperation, rank_recv
        shape = input.shape
        input = input.view(-1)
        min_step = torch.zeros([2,seperation]).to(input.get_device())
        index = torch.zeros(input.shape).type(torch.cuda.LongTensor).to(input.get_device())# need to change type
        src= torch.zeros(input.shape).to(input.get_device())# need to change type
        dist.recv(min_step,rank_recv)
        dist.recv(index,rank_recv)
        dist.recv(src, rank_recv)
        # index = index.view(-1)
        # src = src.view(-1)
        # print("recv_src",src)
        index = index.chunk(seperation)
        src = src.chunk(seperation)
        for i in range(seperation):
            min = min_step[0][i]
            step = min_step[1][i]

            if min == step:
                src[i].index_fill_(0,index[i],min)
                temp = src[i] * 1.0
            else:
                temp = (src[i] + pow(2, ctx.bits - 1)) * step + min
                # print("temp",temp)
                src = torch.cat(src,0)

                src.scatter_(0,index[i],temp)
                src = src.chunk(seperation)  
        src = torch.cat(src,0)
        index = torch.cat(index,0)
        print("rank",1,"min",min,"step",step)
        print("rank",1,"src",src)
        print("rank",1,"index",index)
        input.scatter_(0,index,src)
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx,grad_output):
        bits, seperation, rank_send = ctx.bits, ctx.seperation, ctx.rank_send
        shape = grad_output.shape
        grad_output = grad_output.view(-1)
        src, index = torch.topk(grad_output, int(grad_output.shape[0]))
        index = index.chunk(seperation)
        src = src.chunk(seperation)
        min_step = torch.zeros([2,seperation]).to(grad_output.get_device())
        for i in range(seperation):
            min, max = src[i].min(), src[i].max()
            if min == max:
                min_step[0][i] = min
                min_step[1][i] = min
                temp = src[i]
            else:
                step = (max - min) / (pow(2, bits) - 1)
                min_step[0][i] = min
                min_step[1][i] = step
                temp = torch.round((src[i] - min) / step) - pow(2, bits - 1)
            src = torch.cat(src,0)
            src.scatter_(0,index[i],temp)
            src = src.chunk(seperation)  
        src = torch.cat(src,0)
        index = torch.cat(index,0)
        dist.isend(min_step,rank_send)
        dist.isend(index,rank_send)
        dist.isend(src,rank_send)
        return grad_output.view(shape),None,None,None