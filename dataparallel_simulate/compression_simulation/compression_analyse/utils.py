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


class FakeQuantize(autograd.Function):
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


class SortQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, input, bits, split_bits):
        # print(bits,ratio,partition)
        ctx.bits, ctx.split_bits = bits, split_bits
        shape = input.shape
        test = input
        input = input.view(-1)
        src, index = torch.sort(input, dim=0)
        # print("src_sort",src1,"index1_sort",index1)
        index = torch.tensor_split(index, 2**split_bits)
        src = torch.tensor_split(src, 2**split_bits)
        # print(src1[1])
        for i in range(2**split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)

                # print(torch.round((src1[i] - min) / step)- pow(2, bits - 1) )
                temp_src = torch.round((src[i] - min) / step) - pow(2, bits - 1)

                temp_src = (temp_src + pow(2, bits - 1)) * step + min
                # if i == 0:
                #     print(temp_src)
            else:
                # print(src1[i]
                temp_src = src[i]
            # print("origin_src",src1[i])
            # print("quant_dequant_src",temp_src)

            input.scatter_(0, index[i], temp_src)
        input = input.view(shape)
        return input

    @staticmethod
    def backward(ctx, grad_backward):
        bits, split_bits = ctx.bits, ctx.split_bits
        shape = grad_backward.shape
        src, index = torch.sort(grad_backward, dim=0)
        index = index.chunk(2**split_bits)
        src = src.chunk(2**split_bits)
        for i in range(2**split_bits):
            min, max = src[i].min(), src[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)
                src_temp = torch.round((src[i] - min) / step) - pow(2, bits - 1)
                src_temp = (src_temp + pow(2, bits - 1)) * step + min
            else:
                src_temp = src[i]
            grad_backward.scatter_(0, index[i], src_temp)

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
        return SortQuantization.apply(input, self.bits, self.ratio, self.divide_part)


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
        self.kmeans = KMeans(n_clusters=2**bits, mode="euclidean")
        self.bits = bits

    def forward(self, input):
        return KMeansFunction.apply(input, self.kmeans, self.bits)


class PCAQuantize(autograd.Function):
    @staticmethod
    def forward(ctx, input, q):
        ctx.q = q
        U, S, V = torch.svd_lowrank(input, q=q)
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        output = torch.matmul(output[..., :, :], V[..., :, :])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q = ctx.q
        U, S, V = torch.svd_lowrank(grad_output, q=q)
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        output = torch.matmul(U[..., :, :], S[..., :, :])
        grad_output = torch.matmul(output[..., :, :], V[..., :, :])
        return grad_output, None