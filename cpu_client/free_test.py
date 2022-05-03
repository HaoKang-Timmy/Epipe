"""
Author: your name
Date: 2022-05-02 01:15:06
LastEditTime: 2022-05-02 22:28:46
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/test/dump.py
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import time

input = torch.rand([16, 3, 224, 224])
model = mobilenet_v2(pretrained=True).features[0:1]
last_model = mobilenet_v2(pretrained=True).features[1:].to(0)
# conv2d = torch.nn.Conv2d(3,32,(3,3),stride= (2,2), padding = (1,1),bias= False)
conv2d = model
# output = conv2d(input)

# print(model)
time_sort_avg = 0.0
time_conv_avg = 0.0
cuda_time = 0.0
for i in range(10):
    print(i)
    start = time.time()
    output = conv2d(input)
    time_conv = time.time() - start
    cuda_output = output.to(0)
    torch.cuda.synchronize()
    start = time.time()
    cuda_output = last_model(cuda_output)
    torch.cuda.synchronize()
    time_cuda = time.time() - start

    output = output.view(128, -1)
    # print(output.shape)
    start = time.time()
    sorted, index = torch.sort(output, dim=1)
    # print(sorted)
    time_sort = time.time() - start
    # kth,index =
    time_sort_avg += time_sort
    time_conv_avg += time_conv
    cuda_time += time_cuda
print("sort time avg:", time_sort_avg / 10)
print("conv time avg:", time_conv_avg / 10)
print("cuda time", cuda_time / 10)
