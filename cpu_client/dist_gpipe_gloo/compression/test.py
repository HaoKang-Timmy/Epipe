# from functions import *
# import torch
# import time

# input = torch.rand([64, 32, 112, 112])
# # copy1 = input.clone()
# input_gpu = torch.rand([64, 32, 112, 112]).to(0)
# min_step = torch.rand([2**2, 2])
# min_step_gpu = torch.rand([2**2, 2]).to(0)
# start_cpu = time.time()
# min_step, output = FastQuantization(input, 6, 2, min_step)
# cpu_end = time.time() - start_cpu
# start_cpu = time.time()
# result = FastDequantization(output, 6, 2, min_step, input)
# cpu_end1 = time.time() - start_cpu
# torch.cuda.synchronize(device=0)
# start_gpu = time.time()
# min_step_gpu, output = FastQuantization(input_gpu, 6, 2, min_step_gpu)
# torch.cuda.synchronize(device=0)
# gpu_end = time.time() - start_gpu
# torch.cuda.synchronize(device=0)
# start_gpu = time.time()

# result = FastDequantization(output, 6, 2, min_step_gpu, input_gpu)
# torch.cuda.synchronize(device=0)
# gpu_end1 = time.time() - start_gpu
# torch.cuda.synchronize(device=0)
# start_gpu = time.time()
# min_step_gpu, output = SortQuantization(input_gpu, 6, 2, min_step_gpu)
# torch.cuda.synchronize(device=0)
# gpu_end2 = time.time() - start_gpu
# torch.cuda.synchronize(device=0)
# start_gpu = time.time()
# result = SortDeQuantization(output, 6, 2, min_step_gpu, input_gpu)
# torch.cuda.synchronize(device=0)
# gpu_end3 = time.time() - start_gpu
# print("quantization cpu", cpu_end)
# print("dequantization cpu", cpu_end1)
# print("quantization gpu", gpu_end)
# print("dequantization gpu", gpu_end1)
# print("sq gpu", gpu_end2)
# print("dsq gpu", gpu_end3)
import torch.nn as nn
import torch


class add_tensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p):
        print(p[0])
        p[0] = nn.Parameter(p[0] + 1)

        return p[0]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TestModule(nn.Module):
    def __init__(self) -> None:
        super(TestModule, self).__init__()
        self.p = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0))])

    def forward(self, input):

        return add_tensor.apply(input, self.p)


input = torch.tensor(1.0).to(0)
layer = TestModule().to(0)
print(layer.p[0])
output = layer(input)
print(layer.p[0])
