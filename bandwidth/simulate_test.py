import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import time


class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    if rank == 0:
        # s = torch.cuda.Stream(device=0)
        # with torch.cuda.stream(s):
        input = torch.rand([64, 32, 112, 112]).to(0).type(torch.float16)
        start = time.time()
        dist.send(input, 1)
        end = time.time() - start
        print("send time", end)
        start = time.time()
        dist.isend(input, 1)
        end = time.time() - start
        print("isend time", end)

    else:
        # s = torch.cuda.Stream(device=1)
        # with torch.cuda.stream(s):
        input = torch.rand([64, 32, 112, 112]).to(1).type(torch.float16)
        start = time.time()
        dist.recv(input, 0)
        input = input * 2
        end = time.time() - start
        print("recv time", end)
        start = time.time()
        dist.recv(input, 0)
        input = input * 2
        end = time.time() - start
        print("irecv time", end)

    print(rank)


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))
    pass


if __name__ == "__main__":
    main()
