"""
Author: your name
Date: 2022-04-06 19:44:41
LastEditTime: 2022-04-06 19:45:50
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/bug_kcluster.py
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
import torch
from fast_pytorch_kmeans import KMeans


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


topk_layer = TopkLayer(0.2).to(1)
kmeans = KMeans(n_clusters=2 ** 4, mode="euclidean", device=1)
input = torch.rand([100, 20, 20]).to(1)
output1 = topk_layer(input)
shape = output1.shape
output1 = output1.view(-1, 1)
output2, centers = kmeans.fit_predict(output1)
output2 = output2.view(shape)
print(output2)
print(centers)
