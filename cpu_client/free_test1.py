import torch
import time

input = torch.rand([8, 32, 112, 112])
# shape = input.shape
# input = input.view(64,-1)
# start = time.time()
# kth,index = torch.kthvalue(input[0],10000,dim = 0)
# input = input.view(-1)
# jth,index = torch.kthvalue(input,640000,dim = 0)
# end = time.time()
input = input.view(-1)
shape = input.shape[0]
start = time.time()
src, index = torch.topk(input, int(shape / 4), sorted=False)
end = time.time()
input = input.view(8, -1)
start1 = time.time()
src, index = torch.topk(input, int(shape / 32), sorted=False, dim=1)
end1 = time.time()
start2 = time.time()
kth, index = torch.kthvalue(input[0], int(shape / 32), dim=0)
bool = input > kth
bool = input < kth + 0.5
end2 = time.time()
print(end - start)
print(end1 - start1)
print(end2 - start2)
