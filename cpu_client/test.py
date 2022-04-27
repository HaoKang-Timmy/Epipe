# import torch
# from dist_gpipe_gloo import (
#     SortQuantClient,
#     SortDeQuantClient,
#     SortQuantGPU,
#     SortDeQuantGPU,
# )
# import torch.nn as nn
# import torch.multiprocessing as mp
# import torch.distributed as dist
# import torch.nn.functional as F
# import time


# def main_worker(rank, world_size, args):
#     dist.init_process_group(
#         backend="nccl",
#         init_method="tcp://127.0.0.1:9001",
#         world_size=world_size,
#         rank=rank,
#     )
#     print("process begin", rank)
#     if rank == 0:
#         input = torch.rand([8, 32, 112, 112])
#         for i in range(4):
#             # start =time.time()
#             output = SortQuantClient.apply(input, 6, 2, 1, 0)
#             # print("send over")
#             # end =time.time() -start
#             # print(end)
#     elif rank == 1:
#         input = torch.rand([8, 32, 112, 112]).to(1)
#         for i in range(4):
#             output = SortDeQuantGPU.apply(input, 6, 2, 0)
#     # print(rank,"over")


# def main():
#     mp.spawn(main_worker, nprocs=2, args=(2, 2))


# if __name__ == "__main__":
#     main()

import torch
import time

input = torch.rand([8, 32, 112, 112])
input = input.view(-1)
start = time.time()
src, index = torch.sort(input, dim=0)
end = time.time() - start
print(end)
start = time.time()
src, index = torch.kthvalue(input, 8 * 32 * 112 * 112 - 1)
end = time.time() - start
print(end)
print(src, index)
