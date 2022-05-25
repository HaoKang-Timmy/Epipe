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
import matplotlib.pyplot as plt


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)
    batchsize = 32

    if rank == 0:
        sortquantclient_time = []
        sortdequantclient_time = []
        sortquantserver_time = torch.rand([32]).to(0)
        sortdequantserver_time = torch.rand([32]).to(0)
        i_list = []
        # input = torch.rand([batchsize, 1280, 7, 7])
        # back = torch.rand([10, 10])
        for i in range(batchsize):
            print(i)
            i = i + 1
            i_list.append(i)
            input = torch.rand([i, 32, 112, 112]).requires_grad_()
            back = torch.rand([i, 32, 112, 112])
            start = time.time()
            output = SortQuantClient.apply(input, 6, 2, 1, 0)
            sortquantclient_time.append(time.time() - start)
            start = time.time()
            output.backward(back)
            sortdequantclient_time.append(time.time() - start)
        dist.recv(sortquantserver_time, 1)
        sortquantserver_time = sortquantserver_time.cpu()
        print(sortquantserver_time.shape)
        dist.recv(sortdequantserver_time, 1)
        print(sortdequantserver_time.shape)
        sortdequantserver_time = sortdequantserver_time.cpu()
        l1 = plt.plot(i_list, list(sortquantclient_time), label="sq_client", marker="o")
        l2 = plt.plot(
            i_list, list(sortdequantclient_time), label="sdq_client", marker="o"
        )
        l3 = plt.plot(i_list, list(sortquantserver_time), label="sq_server", marker="o")
        l4 = plt.plot(
            i_list, list(sortdequantserver_time), label="sdq_server", marker="o"
        )
        plt.title("CIFAR10 first layer activation")
        plt.xlabel("batch size")
        plt.ylabel("execution time")
        plt.legend()
        plt.savefig("./test_CIFAR10.jpg")
    elif rank == 1:
        sortquantserver_time = []
        sortdequantserver_time = []
        for i in range(batchsize):
            i = i + 1
            input = torch.rand([i, 32, 112, 112]).requires_grad_().to(1)
            back = torch.rand([i, 32, 112, 112]).to(1)
            # print(input)
            # print(back)
            start = time.time()
            output = SortDeQuantGPU.apply(input, 6, 2, 0)
            sortdequantserver_time.append(time.time() - start)
            start = time.time()
            # print(back.shape)
            output.backward(back)
            sortquantserver_time.append(time.time() - start)
        dist.send(torch.tensor(sortquantserver_time).to(1), 0)
        dist.send(torch.tensor(sortdequantserver_time).to(1), 0)
        # print(torch.tensor(sortdequantserver_time).shape)
        # print(torch.tensor(sortquantserver_time).shape)


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))


if __name__ == "__main__":
    main()
