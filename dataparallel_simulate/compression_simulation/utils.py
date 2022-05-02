"""
Author: your name
Date: 2022-03-26 01:32:34
LastEditTime: 2022-04-08 10:25:58
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/test/utils.py
"""
from cv2 import split
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from fast_pytorch_kmeans import KMeans
import time


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
        ctx, input, bits, min, step, backward_min, backward_step,
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
        # print(bits)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        min, max = grad_output.min(), grad_output.max()
        step = (max - min) / (pow(2, bits) - 1)
        grad_output = torch.round((grad_output - min) / step) - pow(2, bits - 1)
        grad_output = (grad_output + pow(2, bits - 1)) * step + min
        return grad_output, None


class FQBSQ(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, sqbits, split_bits):
        ctx.sqbits, ctx.split_bits = sqbits, split_bits
        min, max = input.min(), input.max()
        step = (max - min) / (pow(2, bits) - 1)
        output = torch.round((input - min) / step) - pow(2, bits - 1)
        output = (output + pow(2, bits - 1)) * step + min
        return output

    @staticmethod
    def backward(ctx, grad_backward):

        bits, split_bits = ctx.sqbits, ctx.split_bits
        shape = grad_backward.shape
        grad_backward = grad_backward.view(-1)
        src, index = torch.sort(grad_backward, dim=0)
        index = torch.tensor_split(index, 2 ** split_bits)
        src = torch.tensor_split(src, 2 ** split_bits)
        for i in range(2 ** split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)
                src_temp = torch.round((src[i] - min) / step) - pow(2, bits - 1)
                src_temp = (src_temp + pow(2, bits - 1)) * step + min
            else:
                src_temp = src[i]
            grad_backward.scatter_(0, index[i], src_temp)

        grad_backward = grad_backward.view(shape)
        return grad_backward, None, None, None


class FSQBQ(autograd.Function):
    def forward(ctx, input, bits, qbits, split_bits):
        shape = input.shape
        input = input.view(-1)
        src, index = torch.sort(input, dim=0)
        index = torch.tensor_split(index, 2 ** split_bits)
        src = torch.tensor_split(src, 2 ** split_bits)
        # print(src1[1])
        for i in range(2 ** split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)

                # print(torch.round((src1[i] - min) / step)- pow(2, bits - 1) )
                temp_src = torch.round((src[i] - min) / step) - pow(2, bits - 1)

                temp_src = (temp_src + pow(2, bits - 1)) * step + min
                # if i == 0:
                #     print(temp_src)
            else:
                # print(src1[i])
                temp_src = src[i]
            # print("origin_src",src1[i])
            # print("quant_dequant_src",temp_src)

            input.scatter_(0, index[i], temp_src)
        ctx.bits = qbits
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        min, max = grad_output.min(), grad_output.max()
        step = (max - min) / (pow(2, bits) - 1)
        grad_output = torch.round((grad_output - min) / step) - pow(2, bits - 1)
        grad_output = (grad_output + pow(2, bits - 1)) * step + min
        return grad_output, None, None, None


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


class SortQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, split_bits):
        # print(bits,ratio,partition)
        shape = input.shape
        input = input.view(-1)
        # print("src_prun",src_temp,"index_prun",index)
        # quantization src1
        # print(src.shape)
        src, index = torch.sort(input, dim=0)
        index = torch.tensor_split(index, 2 ** split_bits)
        src = torch.tensor_split(src, 2 ** split_bits)
        # print(src1[1])
        for i in range(2 ** split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)

                # print(torch.round((src1[i] - min) / step)- pow(2, bits - 1) )
                temp_src = torch.round((src[i] - min) / step) - pow(2, bits - 1)

                temp_src = (temp_src + pow(2, bits - 1)) * step + min
                # if i == 0:
                #     print(temp_src)
            else:
                # print(src1[i])
                temp_src = src[i]
            # print("origin_src",src1[i])
            # print("quant_dequant_src",temp_src)

            input.scatter_(0, index[i], temp_src)

        # src = src.view(-1)
        # index = index.view(-1)
        # print("final_index",index,"final_src",src)
        ctx.bits = bits
        ctx.split_bits = split_bits
        input = input.view(shape)
        # if input.get_device() == 0:
        #     print("forward",torch.abs(torch.abs(input) - torch.abs(test)).sum()/torch.abs(test).sum())
        return input

    @staticmethod
    def backward(ctx, grad_backward):
        bits, split_bits = ctx.bits, ctx.split_bits
        shape = grad_backward.shape
        grad_backward = grad_backward.view(-1)
        src, index = torch.sort(grad_backward, dim=0)
        index = torch.tensor_split(index, 2 ** split_bits)
        src = torch.tensor_split(src, 2 ** split_bits)
        for i in range(2 ** split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)
                src_temp = torch.round((src[i] - min) / step) - pow(2, bits - 1)
                src_temp = (src_temp + pow(2, bits - 1)) * step + min
            else:
                src_temp = src[i]
            grad_backward.scatter_(0, index[i], src_temp)

        grad_backward = grad_backward.view(shape)

        return grad_backward, None, None


# class TopkQuantLayer(nn.Module):
#     def __init__(self, bits, ratio, divide_part):
#         super(TopkQuantLayer, self).__init__()
#         self.bits = bits
#         self.ratio = ratio
#         self.divide_part = divide_part

#     def forward(self, input):
#         return Topk_quantization.apply(input, self.bits, self.ratio, self.divide_part)


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
        for i in range(2 ** bits):
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
        for i in range(2 ** bits):
            labels[labels == i] = centers[i]
        labels = labels.view(shape)
        grad_output = grad_output.view(shape)

        return labels, None, None, None


class KMeansLayer(nn.Module):
    def __init__(self, bits, device) -> None:
        super(KMeansLayer, self).__init__()
        self.kmeans = KMeans(n_clusters=2 ** bits, mode="euclidean", device=device)
        self.bits = bits

    def forward(self, input):
        return KMeansFunction.apply(input, self.kmeans, self.bits)


class PCAQuantize(autograd.Function):
    @staticmethod
    def forward(ctx, input, q):
        ctx.q = q
        # device = input.get_device()
        # input = input.cpu()
        # start = time.time()
        U, S, V = torch.svd_lowrank(input, q=q)
        # print("svd time",time.time() - start)
        # U = U.to(device)
        # S = S.to(device)
        # V = V.to(device)
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)

        output = torch.matmul(U[..., :, :], S[..., :, :])
        output = torch.matmul(output[..., :, :], V[..., :, :])
        # print("pca en de",time.time() - start)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q = ctx.q
        device = grad_output.get_device()
        # grad_output = grad_output.cpu()
        U, S, V = torch.svd_lowrank(grad_output, q=q)
        # U = U.to(device)
        # S = S.to(device)
        # V = V.to(device)
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)

        output = torch.matmul(U[..., :, :], S[..., :, :])
        grad_output = torch.matmul(output[..., :, :], V[..., :, :])
        return grad_output, None


# class fastpca(autograd.Function):
#     @staticmethod
#     def forward(ctx,input,q):
#         shape = input.shape
#         input = input.view(-1,shape[-2],shape[-1])
#         pass


# import torch.nn as nn
# import torch


class nlp_sequential(nn.Module):
    def __init__(self, layers: list):
        super(nlp_sequential, self).__init__()
        self.layers = layers[0]

    def forward(self, output: torch.tensor, mask: torch.tensor):
        for i, layer in enumerate(self.layers):
            output = layer(output, mask)
            output = output[0]
        return output


class combine_embeding(nn.Module):
    def __init__(self, layers: list, embed_layer):
        super(combine_embeding, self).__init__()
        self.layers = layers[0]
        self.embed_layer = embed_layer[0]

    def forward(self, input: torch.tensor, mask: torch.tensor):
        output = self.embed_layer(input)

        output = self.layers(output, mask)
        output = output
        return output


class combine_classifier(nn.Module):
    def __init__(self, layers: list, classifier):
        super(combine_classifier, self).__init__()
        self.layers = layers[0]
        self.classifier = classifier[0]

    def forward(self, output: torch.tensor, mask: torch.tensor):
        # for i, layer in enumerate(self.layers):
        output = self.layers(output, mask)
        output = output
        output = self.classifier(output)
        return output


<<<<<<< Updated upstream
class FSVDBSQ(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, bits, split_bits):
        ctx.bits, ctx.split_bits = bits, split_bits
        device = input.get_device()
        input = input.cpu()
        U, S, V = torch.svd_lowrank(input, q=q)
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        input = torch.matmul(output[..., :, :], V[..., :, :])
        input = input.to(device)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits, split_bits = ctx.bits, ctx.split_bits
        shape = grad_output.shape
        grad_output = grad_output.view(-1)
        src, index = torch.sort(grad_output, dim=0)
        index = torch.tensor_split(index, 2 ** split_bits)
        src = torch.tensor_split(src, 2 ** split_bits)
        for i in range(2 ** split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)
                src_temp = torch.round((src[i] - min) / step) - pow(2, bits - 1)
                src_temp = (src_temp + pow(2, bits - 1)) * step + min
            else:
                src_temp = src[i]
            grad_output.scatter_(0, index[i], src_temp)

        grad_output = grad_output.view(shape)

        return grad_output, None, None, None


class FSQBSVD(autograd.Function):
    @staticmethod
    def forward(ctx, input, q, bits, split_bits):
        ctx.q = q
        shape = input.shape
        input = input.view(-1)
        src, index = torch.sort(input, dim=0)
        index = torch.tensor_split(index, 2 ** split_bits)
        src = torch.tensor_split(src, 2 ** split_bits)
        # print(src1[1])
        for i in range(2 ** split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)
                temp_src = torch.round((src[i] - min) / step) - pow(2, bits - 1)

                temp_src = (temp_src + pow(2, bits - 1)) * step + min
            else:
                temp_src = src[i]
            input.scatter_(0, index[i], temp_src)
        ctx.bits = bits
        ctx.split_bits = split_bits
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        q = ctx.q
        device = grad_output.get_device()
        grad_output = grad_output.cpu()
        U, S, V = torch.svd_lowrank(grad_output, q=q)
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        grad_output = torch.matmul(output[..., :, :], V[..., :, :])
        grad_output = grad_output.to(device)
        return grad_output


class PartialQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, partial_bits):
        max, min = input.max(), input.min()
=======
class PartialQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, split_bits):
        shape = input.shape
        input = input.view(-1)
        min, max = input.min(), input.max()
        medium = (min + max) / 2
        greater = torch.numel(input[input > medium])
        rate = greater / torch.numel(input)
        right_bits = torch.ceil((2 ** split_bits) * right_bits)
        left_bits = torch.ceil((2 ** split_bits) * right_bits)


# class LoraLinear(autograd.Function):
#     @staticmethod
#     def forward(ctx, input,weight):
#         ctx.save_for_backward(input,weight)
#         output = torch.matmul(input,weight)
#         return output
#     @staticmethod
#     def backward(ctx,grad_output):
#         input,weight =
>>>>>>> Stashed changes
