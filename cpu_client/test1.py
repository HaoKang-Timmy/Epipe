import torch
import time

input = torch.randn([8, 32, 112, 112])
input = input.view(-1)
start = time.time()
src, index = torch.sort(input)
end = time.time() - start
print(end)
start = time.time()
src, index = torch.topk(input, k=2 * 32 * 112 * 112, sorted=False)
print(time.time() - start)
x = torch.zeros((2, 2), dtype=torch.long)
print(x)
zero = torch.tensor(0)
zero.type(torch.long)
one = torch.tensor(1).type(torch.long)
minus = torch.tensor(-1).type(torch.long)
print(torch.where((x < one) & (x > minus), x, 0))
