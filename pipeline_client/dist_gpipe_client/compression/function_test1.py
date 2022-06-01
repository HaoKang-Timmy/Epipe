from functions import *
import torch
import time


def error(input, label):
    difference = torch.abs(input) - torch.abs(label)
    return torch.abs(difference).mean()


rank = 3
input = torch.rand([64, 32, 112, 112])
p_buffer = [torch.rand([64, 112 * 112, rank])]
q_buffer = [torch.rand([64, 32, rank])]
p, q = PowerSVD(input, q_buffer, p_buffer, 2)
output = PowerSVDDecompress(p, q, input.shape)
print(error(input, output))
