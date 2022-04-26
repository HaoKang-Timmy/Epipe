import torch.multiprocessing as mp
from compression_layer_nccl import PCASendClient,PCARecvGPU
import torch.distributed as dist
import torch


def main():
    mp.spawn(main_worker, nprocs=2, args=(4, 4))


def main_worker(rank, nothing, nothing1):

    torch.backends.cudnn.benckmark = False
    print(rank)
    if rank == 0:
        dist.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:10000", world_size=2, rank=rank
        )
        
    elif rank == 1:
        dist.init_process_group(
            backend="gloo", init_method="tcp://127.0.0.1:10000", world_size=2, rank=rank
        )
       


if __name__ == "__main__":
    main()
