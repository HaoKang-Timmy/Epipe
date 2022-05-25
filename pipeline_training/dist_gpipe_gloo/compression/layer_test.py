import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time

from compression_layer_nccl import (
    PowerSVDClientSendLayer,
    PowerSVDServerRecvLayer,
    PowerSVDClientRecvLayer,
    PowerSVDServerSendLayer,
)


def error(input, label):
    difference = torch.abs(input) - torch.abs(label)
    return torch.abs(difference).mean()


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)

    if rank == 0:
        input = torch.rand([64, 32, 112, 112])
        layer = PowerSVDClientRecvLayer(3, input.shape, 2, 0, 1)
        output = layer(input)
        output = output.to(0)
        dist.send(output, 1)
    elif rank == 1:
        input = torch.rand([64, 32, 112, 112]).to(1)
        recv = torch.rand([64, 32, 112, 112]).to(1)
        layer = PowerSVDServerSendLayer(3, input.shape, 2, 0, 1).to(1)
        output = layer(input)
        dist.recv(recv, 0)
        print(error(output, recv))


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))


if __name__ == "__main__":
    main()
