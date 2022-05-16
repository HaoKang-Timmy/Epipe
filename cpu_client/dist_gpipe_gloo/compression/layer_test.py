import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)

    if rank == 0:
        q_buffer = torch.rand([2, 3, 4]).to(0)
        Q, R = torch.linalg.qr(q_buffer)
        Q = Q * 1
        print("local before q_buffer", Q, Q.shape)

        dist.send(Q, 1)

    elif rank == 1:

        q_buffer = torch.rand([2, 3, 3]).to(1)

        dist.recv(q_buffer, 0)

        print("recv", q_buffer, q_buffer.shape)


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))


if __name__ == "__main__":
    main()
