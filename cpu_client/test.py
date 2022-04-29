import torch
from dist_gpipe_gloo import (
    SortQuantClient,
    SortDeQuantClient,
    SortQuantGPU,
    SortDeQuantGPU,
)
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import time


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)
    if rank == 0:
        input = torch.rand([8, 1280, 7, 7])
        back = torch.rand([10, 10])
        # print(input)
        # for i in range(4):
        # start =time.time()
        # quant_time = time.time()
        # output = SortQuantClient.apply(input, 6, 2, 1, 0)
        # end_time = time.time() - quant_time
        # print(end_time)

        dequant_time = time.time()
        output = SortDeQuantClient.apply(input, 6, 2, 1, 0)
        end_time = time.time() - dequant_time
        print(end_time)
        # print(end_time)
        # print("rank0" ,output)
        # print("send over")
        # end =time.time() -start
        # print(end)
    elif rank == 1:
        input = torch.rand([8, 32, 112, 112]).to(1)
        # for i in range(4):
        # output = SortDeQuantGPU.apply(input, 6, 2, 0)

        output = SortQuantGPU.apply(input, 6, 2, 0)
        # print("rank1" ,output)
        # print(output)
    # print(rank,"over")


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))


if __name__ == "__main__":
    main()
