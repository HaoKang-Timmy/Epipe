import torch
from compression_layer_nccl import (
    FastQuantClient,
    FastDequantizationGPU,
    FastDequantizationServer,
    FastDequantClient,
    FastQuantizationServer,
)
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


def error(input, label):
    difference = input - label


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)

    if rank == 0:
        for i in range(10):
            input = torch.rand([64, 32, 112, 112]).requires_grad_()
            recv = torch.zeros([64, 32, 112, 112])
            # input = torch.rand([10,10])
            # send[0,0] = 1e-9
            # send[3,5] = 2e-9
            # print(input)
            torch.cuda.synchronize()
            start = time.time()
            output = FastDequantClient.apply(input, 6, 2, 1, 0)
            torch.cuda.synchronize()
            end = time.time()
            print("rank0 forward", end - start)
            torch.cuda.synchronize()
            start = time.time()
            output.backward(recv)
            torch.cuda.synchronize()
            end = time.time()
            print("rank0 backward", end - start)
            # print(recv)
    elif rank == 1:
        # layer = FastDequantizationServerLayer(6,2,0).to(1)
        for i in range(10):
            input = torch.rand([64, 32, 112, 112]).requires_grad_().to(1)
            send = torch.zeros([64, 32, 112, 112]).to(1)
            # input[0,0] = 1e-9
            # input[3,5] = 2e-9
            # input = torch.rand([10,10]).to(1)
            torch.cuda.synchronize()
            start = time.time()
            output = FastQuantizationServer.apply(input, 6, 2, 0)
            torch.cuda.synchronize()
            end = time.time()
            print("rank1 forward", end - start)
            torch.cuda.synchronize()
            start = time.time()
            output.backward(send)
            torch.cuda.synchronize()
            end = time.time()
            print("rank1 backward", end - start)
            # print(output)


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))


if __name__ == "__main__":
    main()
