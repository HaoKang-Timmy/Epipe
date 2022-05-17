import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time


def PowerSVD(input: torch.tensor, q_buffer: list, p_buffer: list, n_iter):
    shape = input.shape
    input = input.view(int(input.shape[0]), int(input.shape[1]), -1)
    for i in range(n_iter):
        if i == n_iter - 1:
            p_buffer[0] = torch.linalg.qr(p_buffer[0]).Q
        q_buffer[0] = input @ p_buffer[0]
        if i == n_iter - 1:
            q_buffer[0] = torch.linalg.qr(q_buffer[0]).Q
        p_buffer[0] = input.permute((0, 2, 1)) @ q_buffer[0]
    input = input.view(shape)


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)

    if rank == 0:
        input = torch.rand([2, 4, 4]).to(0)
        p_buffer = [torch.rand([2, 4, 3]).to(0)]
        q_buffer = [torch.rand([2, 4, 3]).to(0)]
        # Q, R = torch.linalg.qr(q_buffer)
        PowerSVD(input, q_buffer, p_buffer, 2)
        print("send p ", p_buffer[0])
        print("send q ", q_buffer[0])
        dist.send(p_buffer[0], 1)
        dist.send(q_buffer[0], 1)
    elif rank == 1:
        p_buffer = torch.rand([2, 4, 3]).to(1)
        q_buffer = torch.rand([2, 4, 3]).to(1)
        dist.recv(p_buffer, 0)
        dist.recv(q_buffer, 0)
        print("recv p", p_buffer)
        print("recv q", q_buffer)


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))


if __name__ == "__main__":
    main()
