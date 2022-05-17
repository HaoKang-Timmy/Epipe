from functions import *
import torch
import time

input = torch.rand([64, 32, 112, 112])
# copy1 = input.clone()
input_gpu = torch.rand([64, 32, 112, 112]).to(0)
min_step = torch.rand([2 ** 2, 2])
min_step_gpu = torch.rand([2 ** 2, 2]).to(0)
start_cpu = time.time()
min_step, output = FastQuantization(input, 6, 2, min_step)
cpu_end = time.time() - start_cpu
start_cpu = time.time()
result = FastDequantization(output, 6, 2, min_step, input)
cpu_end1 = time.time() - start_cpu
torch.cuda.synchronize(device=0)
start_gpu = time.time()
min_step_gpu, output = FastQuantization(input_gpu, 6, 2, min_step_gpu)
torch.cuda.synchronize(device=0)
gpu_end = time.time() - start_gpu
torch.cuda.synchronize(device=0)
start_gpu = time.time()

result = FastDequantization(output, 6, 2, min_step_gpu, input_gpu)
torch.cuda.synchronize(device=0)
gpu_end1 = time.time() - start_gpu
torch.cuda.synchronize(device=0)
start_gpu = time.time()
min_step_gpu, output = SortQuantization(input_gpu, 6, 2, min_step_gpu)
torch.cuda.synchronize(device=0)
gpu_end2 = time.time() - start_gpu
torch.cuda.synchronize(device=0)
start_gpu = time.time()
result = SortDeQuantization(output, 6, 2, min_step_gpu, input_gpu)
torch.cuda.synchronize(device=0)
gpu_end3 = time.time() - start_gpu
print("quantization cpu", cpu_end)
print("dequantization cpu", cpu_end1)
print("quantization gpu", gpu_end)
print("dequantization gpu", gpu_end1)
print("sq gpu", gpu_end2)
print("dsq gpu", gpu_end3)
