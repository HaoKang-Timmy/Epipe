import torch.multiprocessing as mp
from compression_layer_gloo import Qsend, Qrecv
import torch.distributed as dist
import torch


def main():
    mp.spawn(main_worker, nprocs=2, args=(4, 4))


def main_worker(rank, nothing, nothing1):

    torch.backends.cudnn.benckmark = False
    print(rank)
    if rank == 0:
        dist.init_process_group(
            backend="gloo", init_method="tcp://127.0.0.1:10000", world_size=2, rank=rank
        )
        for i in range(2):
            input = torch.rand([10, 10]).requires_grad_()
            # output = Qsend.apply(input,8,1)
            dist.isend(input, 1)
            print("rank0", input)
    elif rank == 1:
        dist.init_process_group(
            backend="gloo", init_method="tcp://127.0.0.1:10000", world_size=2, rank=rank
        )
        # input = torch.rand([10,10]).to(1)
        # dist.recv(input,0)
        for i in range(2):

            input = torch.rand([10, 10]).requires_grad_()
            # output = Qrecv.apply(input, 8,0)
            dist.recv(input, 0)
            print("rank1", input)


if __name__ == "__main__":
    main()
