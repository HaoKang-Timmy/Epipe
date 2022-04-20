"""
Author: your name
Date: 2022-03-26 01:32:34
LastEditTime: 2022-04-08 10:25:58
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/test/utils.py
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from fast_pytorch_kmeans import KMeans


def relative_error(origin, quant):
    return torch.mean(
        torch.abs((torch.abs(origin) - torch.abs(quant))) / torch.abs(origin)
    )


def error(origin, quant):
    return torch.abs(torch.abs(origin) - torch.abs(quant)).sum()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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


class Quantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, min, step, bits, backward_min, backward_step):
        ctx.bits, ctx.backward_min, ctx.backward_step = (
            bits,
            backward_min,
            backward_step,
        )
        # if bits <= 8:
        #     output = torch.round((input - min) / step - pow(2, bits - 1)).type(
        #         torch.cuda.CharTensor
        #     )
        # else:

        output = torch.round((input - min) / step) - pow(2, bits - 1)  # 16
        # pass
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bits, backward_min, backward_step = (
            ctx.bits,
            ctx.backward_min[0],
            ctx.backward_step[0],
        )
        output = (grad_output + pow(2, bits - 1)) * backward_step + backward_min
        return output, None, None, None, None, None


class QuantizationLayer(nn.Module):
    def __init__(self, bits):
        super(QuantizationLayer, self).__init__()

        self.bits = bits
        self.backward_step = torch.tensor([0.0])
        self.backward_min = torch.tensor([0.0])

    def forward(self, x):
        min, max = x.min(), x.max()
        step = (max - min) / (pow(2, self.bits) - 1)  # error
        # min = torch.tensor([min.item()]).to(x.get_device())
        # print("steps",step,"minus",max-min,"results",(max - min) / (pow(2, self.bits)-1),self.bits)

        return (
            Quantization.apply(
                x, min, step, self.bits, self.backward_min, self.backward_step
            ),
            min,
            step,
        )


class Dequantization(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        bits,
        min,
        step,
        backward_min,
        backward_step,
    ):
        ctx.bits, ctx.backward_min, ctx.backward_step = (
            bits,
            backward_min,
            backward_step,
        )
        # print("input",input,(input + pow(2, bits - 1)))
        output = (input + pow(2, bits - 1)) * step + min
        # print("output",output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits

        min, max = grad_output.min(), grad_output.max()
        step = (max - min) / (pow(2, bits) - 1)
        ctx.backward_min[0] = min
        ctx.backward_step[0] = step
        output1 = torch.round((grad_output - min) / step) - pow(2, bits - 1)

        return output1, None, None, None, None, None


class DequantizationLayer(nn.Module):
    def __init__(self, bits):
        super(DequantizationLayer, self).__init__()
        self.bits = bits

    def forward(self, input, min, step, backward_min, backward_step):

        return Dequantization.apply(
            input, self.bits, min, step, backward_min, backward_step
        )


class Fakequantize(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits):
        ctx.bits = bits
        min, max = input.min(), input.max()
        step = (max - min) / (pow(2, bits) - 1)
        output = torch.round((input - min) / step) - pow(2, bits - 1)
        output = (output + pow(2, bits - 1)) * step + min
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        min, max = grad_output.min(), grad_output.max()
        step = (max - min) / (pow(2, bits) - 1)
        grad_output = torch.round((grad_output - min) / step) - pow(2, bits - 1)
        grad_output = (grad_output + pow(2, bits - 1)) * step + min
        return grad_output, None


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


class Topk_quantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, ratio, partition):
        # print(bits,ratio,partition)
        shape = input.shape
        test = input
        input = input.view(-1)

        mask = torch.zeros(input.shape).to(input.get_device())
        src, index = torch.topk(torch.abs(input), int(ratio * input.shape[0]))

        mask.index_fill_(0, index, 1.0)
        input = input * mask
        index = input.nonzero()
        index = index.view(-1)
        src = input.index_select(0, index)
        # print("src_prun",src_temp,"index_prun",index)
        # quantization src1
        # print(src.shape)
        src1, index1 = torch.topk(src, int(src.shape[0]))
        # print("src_sort",src1,"index1_sort",index1)
        index1 = index1.chunk(partition)
        src1 = src1.chunk(partition)
        # print(src1[1])
        for i in range(partition):
            min, max = src1[i].min(), src1[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)

                # print(torch.round((src1[i] - min) / step)- pow(2, bits - 1) )
                temp_src = torch.round((src1[i] - min) / step) - pow(2, bits - 1)

                temp_src = (temp_src + pow(2, bits - 1)) * step + min
                # if i == 0:
                #     print(temp_src)
            else:
                # print(src1[i])
                temp_src = src1[i]
            # print("origin_src",src1[i])
            # print("quant_dequant_src",temp_src)

            src.scatter_(0, index1[i], temp_src)

        # src = src.view(-1)
        # index = index.view(-1)
        # print("final_index",index,"final_src",src)
        input.scatter_(0, index, src)
        ctx.mask = mask.view(shape)
        ctx.ratio = ratio
        ctx.bits = bits
        ctx.partition = partition
        input = input.view(shape)
        # if input.get_device() == 0:
        #     print("forward",torch.abs(torch.abs(input) - torch.abs(test)).sum()/torch.abs(test).sum())
        return input

    @staticmethod
    def backward(ctx, grad_backward):
        test = grad_backward
        shape = grad_backward.shape
        grad_backward = grad_backward * ctx.mask
        grad_backward = grad_backward.view(-1)
        index = grad_backward.nonzero()
        index = index.view(-1)
        src = grad_backward.index_select(0, index)
        src = src.view(-1)
        src1, index1 = torch.topk(src, int(src.shape[0]))
        index1 = index1.chunk(ctx.partition)
        src1 = src1.chunk(ctx.partition)
        for i in range(ctx.partition):
            min, max = src1[i].min(), src1[i].max()
            if min != max:
                step = (max - min) / (pow(2, ctx.bits) - 1)
                src_temp = torch.round((src1[i] - min) / step) - pow(2, ctx.bits - 1)
                src_temp = (src_temp + pow(2, ctx.bits - 1)) * step + min
            else:
                src_temp = src1[i]
            src.scatter_(0, index1[i], src_temp)

        # index = index.view(-1)
        grad_backward.scatter_(0, index, src)
        grad_backward = grad_backward.view(shape)
        # print(grad_backward)
        # while(1):
        #     pass
        # if grad_backward.get_device() == 0:
        #     print("backward",torch.abs(torch.abs(grad_backward) - torch.abs(test)).sum()/torch.abs(test).sum())
        return grad_backward, None, None, None


class TopkQuantLayer(nn.Module):
    def __init__(self, bits, ratio, divide_part):
        super(TopkQuantLayer, self).__init__()
        self.bits = bits
        self.ratio = ratio
        self.divide_part = divide_part

    def forward(self, input):
        return Topk_quantization.apply(input, self.bits, self.ratio, self.divide_part)


class KMeansFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, kmeans, bits, settings=0):
        ctx.kmeans = kmeans
        # ctx.settings = settings
        ctx.bits = bits
        shape = input.shape

        input = input.view(-1, 1)
        labels, centers = kmeans.fit_predict(input)
        centers = centers.view(-1)
        labels = labels.type(torch.cuda.FloatTensor)
        for i in range(2**bits):
            labels[labels == i] = centers[i]
        labels = labels.view(shape)
        labels = labels.requires_grad_()
        return labels

    @staticmethod
    def backward(ctx, grad_output):
        kmeans, bits = ctx.kmeans, ctx.bits
        shape = grad_output.shape
        # settings = ctx.settings
        grad_output = grad_output.view(-1, 1)
        labels, centers = kmeans.fit_predict(grad_output)
        centers = centers.view(-1)
        labels = labels.type(torch.cuda.FloatTensor)
        for i in range(2**bits):
            labels[labels == i] = centers[i]
        labels = labels.view(shape)
        grad_output = grad_output.view(shape)

        return labels, None, None, None


class KMeansLayer(nn.Module):
    def __init__(self, bits, device) -> None:
        super(KMeansLayer, self).__init__()
        self.kmeans = KMeans(n_clusters=2**bits, mode="euclidean", device=device)
        self.bits = bits

    def forward(self, input):
        return KMeansFunction.apply(input, self.kmeans, self.bits)

class PCAQuantize(autograd.Function):
    @staticmethod
    def forward(ctx,input,q):
        ctx.q = q
        U,S,V = torch.pca_lowrank(input,q = q)
        S = torch.diag_embed(S)
        output = torch.matmul(U[...,:,:],S[...,:,:])
        output = torch.matmul(output[...,:,:],V[...,:,:])
        return output
    @staticmethod
    def backward(ctx,grad_output):
        q = ctx.q
        U,S,V = torch.pca_lowrank(grad_output,q = q)
        S = torch.diag_embed(S)
        output = torch.matmul(U[...,:,:],S[...,:,:])
        grad_output = torch.matmul(output[...,:,:],V[...,:,:])
        return grad_output
        

