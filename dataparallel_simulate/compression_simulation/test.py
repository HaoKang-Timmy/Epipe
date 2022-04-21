import torch
from utils import SortQuantization

input = torch.rand([10, 10])
print(input)
output = SortQuantization.apply(input, 6, 2)
print(output)
