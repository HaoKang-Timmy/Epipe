from functions import *
import torch
import time

input = torch.zeros([8, 32, 112, 112])
# input[0,0] = 0.0
# print(input)
recever = torch.rand([8, 32, 112, 112])
min_step = torch.zeros([2**2, 4])
other = input.view(-1)
start = time.time()
min_step, output = FastQuantizationCPU(input, 6, 2, min_step)
end = time.time()
print(end - start)
# print(input)
start = time.time()
recever = FastDequantizationCPU(output, 6, 2, min_step, recever)
# print(recever)
end = time.time()
print(end - start)
start = time.time()
input = torch.rand([8, 1280, 7, 7])
some = torch.svd_lowrank(input, q=2)
end = time.time()
print(end - start)


# print(recever)
