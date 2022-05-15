import torch
from compression_layer_nccl import (
    FastQuantClient,
    FastDequantizationServer,
    # FastDequantizationServer,
    FastDequantClient,
    FastQuantizationServer,
    PowerSVDServerRecvLayer,
    QSendClient,
    QrecvGPU,
    PowerSVDClientRecvLayer,
    PowerSVDServerSendLayer,
)
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import time


# from gpipe_test.cpu_client.dist_gpipe_gloo.compression.functions import FastQuantization


def error(input, label):
    difference = torch.abs(input) - torch.abs(label)
    # print(input.shape)
    # print(label.shape)
    # print(input)
    # print(label)
    return torch.abs(difference).mean(), torch.abs(difference).max()


def main_worker(rank, world_size, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:9001",
        world_size=world_size,
        rank=rank,
    )
    print("process begin", rank)

    if rank == 0:
        for i in range(1):
            s = torch.cuda.Stream(device=0)
            with torch.cuda.stream(s):
                input = torch.rand([64, 32, 112, 112]).requires_grad_()
                recv = torch.rand([64, 32, 112, 112])
                # recv = torch.zeros([64, 32, 112, 112]).to(0)
                # input = torch.rand([10,10])
                # send[0,0] = 1e-9
                # send[3,5] = 2e-9
                # print(input)
                # torch.cuda.synchronize()
                # start = time.time()
                layer1 = PowerSVDClientRecvLayer(1, input.shape, 2, 0, 1)
                output = layer1(input)
                output.backward(recv)
                # torch.cuda.synchronize()
                # end = time.time()
                # output = output.to(0)
                # dist.recv(recv, 1)
                # print("recv origin", recv[0])
                # print("recv dequant", output[0])
                # print(error(recv, output))
    elif rank == 1:
        s = torch.cuda.Stream(device=1)
        with torch.cuda.stream(s):
            # layer = FastDequantizationServerLayer(6,2,0).to(1)
            for i in range(1):
                input = torch.rand([64, 32, 112, 112]).requires_grad_().to(1)
                some = input.clone().detach()
                layer2 = PowerSVDServerSendLayer(1, input.shape, 2, 0).to(1)
                output = layer2(input)
                output.backward(some)
    print("finish")


def main():
    mp.spawn(main_worker, nprocs=2, args=(2, 2))


if __name__ == "__main__":
    main()
