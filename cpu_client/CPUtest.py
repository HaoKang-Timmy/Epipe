"""
Author: your name
Date: 2022-05-01 13:29:23
LastEditTime: 2022-05-01 13:29:44
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/cpu_client/CPUtest.py
"""
"""
Author: your name
Date: 2022-04-26 01:38:00
LastEditTime: 2022-04-30 01:19:35
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/cpu_client/CPUtest.py
"""
from torchvision.models import mobilenet_v2
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

model = mobilenet_v2(pretrained=True)
model.classifier[-1] = nn.Linear(1280, 10)
layer1 = [model.features[0:1]]
layer1 = nn.Sequential(*layer1)
# input = torch.rand([1,3,224,224])
# output = layer1(input)
# print(output.shape)
sq_list = []
svd_list = []
lowrank_list = []
i_list = []
first_layer_list = []
for i in range(32):
    i = i + 1

    input = torch.rand([i, 32, 112, 112])
    start = time.time()
    output = layer1(input)
    first_layer_time = time.time() - start

    start = time.time()
    U, S, V = torch.svd_lowrank(input, q=3)
    print(U)
    print(S)
    print(V)

    while 1:
        pass
    lowrank_time = time.time() - start
    start = time.time()
    U, S, V = torch.svd(input)
    svd_time = time.time() - start
    start = time.time()
    output = input.view(-1)
    output = torch.sort(output, dim=0)
    sortquant_time = time.time() - start

    sq_list.append(sortquant_time)
    svd_list.append(svd_time)
    lowrank_list.append(lowrank_time)
    i_list.append(i)
    print(i)
l1 = plt.plot(i_list, sq_list, label="sortquant", marker="o")
l2 = plt.plot(i_list, lowrank_list, label="svd_lowrank", marker="o")
l3 = plt.plot(i_list, svd_list, label="svd", marker="o")
plt.title("RTE activation memory execution tests")
plt.xlabel("batch size")
plt.ylabel("execution time")
plt.legend()
plt.savefig("./test_cpu_rte.jpg")
