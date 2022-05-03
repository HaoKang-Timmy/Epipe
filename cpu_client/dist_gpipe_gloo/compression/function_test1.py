from functions import FastDeQuantization, FastQuantization
import torch
import time

input = torch.rand([10, 15]).to(0)
# input[0,0] = 0.0
# print(input)
recever = torch.rand([10, 15]).to(0)
min_step = torch.zeros([2 ** 2, 2]).to(0)
other = input.view(-1)
start = time.time()
some, index = torch.sort(other)
end = time.time()
print(end - start)
start = time.time()
min_step, output = FastQuantization(input, 6, 2, min_step)
end = time.time()
print(end - start)
start = time.time()
recever = FastDeQuantization(output, 6, 2, min_step, recever)
# print(recever)
end = time.time()
print(end - start)
