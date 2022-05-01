"""
Author: your name
Date: 2022-05-01 13:29:23
LastEditTime: 2022-05-01 19:33:29
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/cpu_client/free_test.py
"""
import torch
import time
import torchsort

# from torchvision.models import mobilenet_v2
input1 = torch.rand([1, 10000000])
# input1 = input1.view(-1)
# input1 = input1.view(1,-1)
print(input1.shape)
start = time.time()
over = torchsort.soft_sort(input1, regularization_strength=0.1)
print(time.time() - start)
# start = time.time()
# U,S,V = torch.svd_lowrank(input1, q = 2)
# print(time.time() - start)

# model = mobilenet_v2(pretrained=True)
# model = model.features[0]
# # inputs
# input = torch.rand([32, 3, 224, 224])
# start = time.time()
# input = model(input)
# print(time.time() - start)
# input = input.view(-1)
# start = time.time()
# medium = torch.median(input)
# print(time.time() - start)
# start = time.time()
# print(input.shape[0])
# medium = torch.kthvalue(input, (input.shape[0]))
# print(time.time() - start)
# start = time.time()
# src, index = torch.sort(input)
# print(time.time() - start)
