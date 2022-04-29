import torch.multiprocessing as mp
from compression_layer_nccl import (
    PCASendClient,
    PCARecvGPU,
    CompressionClientRecv,
    CompressionClientSend,
    CompressRecvGPU,
    CompressSendGPU,
)
import torch.distributed as dist
import torch
import time


def main():
    mp.spawn(main_worker, nprocs=2, args=(4, 4))


def main_worker(rank, nothing, nothing1):

    torch.backends.cudnn.benckmark = False
    print(rank)
    if rank == 0:
        dist.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:10000", world_size=2, rank=rank
        )

        input = torch.rand([8, 1280, 32, 32]).requires_grad_()
        back = torch.rand([8, 1280, 32, 32])
        for i in range(10):
            start = time.time()
            output = CompressionClientSend.apply(input, 3, 1, 0, 6, 2)
            print("client forward", time.time() - start)
            start = time.time()
            output.backward(back)
            # print("client backward",time.time() - start)
    elif rank == 1:
        dist.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:10000", world_size=2, rank=rank
        )
        input = torch.rand([8, 1280, 32, 32]).to(1).requires_grad_()
        back = torch.rand([8, 1280, 32, 32]).to(1)
        for i in range(10):
            start = time.time()
            output = CompressRecvGPU.apply(input, 3, 0, 6, 2)
            print("server forward", time.time() - start)
            start = time.time()
            output.backward(back)
            # print("server backward",time.time() - start)


if __name__ == "__main__":
    main()
