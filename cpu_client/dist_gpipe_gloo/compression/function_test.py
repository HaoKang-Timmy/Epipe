import torch
from functions import *
import time

split_bits = 2
min_step = torch.rand([2 ** split_bits, 2])
input = torch.rand([10, 10])
print("before quant")
print(input)
min_step, recv = SortQuantization(input, 7, 2, min_step)
output1 = torch.rand([10, 10]).view(-1)
# output1 = SortDeQuantization(recv,6,2,min_step,output1)
# print("after quant")
# print(recv)
recv = recv.view(-1)
# start = time.time()
output1 = FastDeQuantization(recv, 7, 2, min_step, output1)
# print(time.time() - start)
output1 = output1.view(10, 10)
print(output1)
