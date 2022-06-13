from torchvision.models import mobilenet_v2
import torch.nn as nn
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import time
import torch.distributed as dist
import torch.nn.functional as F


class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def main():
    chunk = 4
    mp.spawn(main_worker, nprocs=2, args=(2, chunk))


def main_worker(rank, nprocess, chunk):
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:1234", world_size=2, rank=rank,
    )
    model = mobilenet_v2(pretrained=True)
    model.classifier[-1] = nn.Linear(1280, 10)
    feature = model.features[0].children()
    conv = next(feature)
    bn = next(feature)
    conv1 = nn.Conv2d(32, 32, (4, 4), (4, 4))
    conv2 = nn.Conv2d(32, 16, (1, 1), (1, 1))
    tconv_2 = nn.Conv2d(16, 32, (1, 1), (1, 1))
    tconv_1 = nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
    conv_3 = nn.Conv2d(1280, 320, (1, 1), (1, 1))
    conv4 = nn.Conv2d(320, 100, (1, 1), (1, 1))
    tconv_4 = nn.ConvTranspose2d(100, 320, (1, 1), (1, 1))
    tconv_3 = nn.ConvTranspose2d(320, 1280, (1, 1), (1, 1))

    layer1 = [conv, bn, conv1, conv2]
    layer2 = [
        tconv_2,
        tconv_1,
        nn.ReLU6(inplace=False),
        model.features[1:],
        conv_3,
        conv4,
    ]
    layer3 = [tconv_4, tconv_3, Reshape1(), model.classifier]

    layer1 = nn.Sequential(*layer1)
    layer2 = nn.Sequential(*layer2)
    layer3 = nn.Sequential(*layer3)
    recv = torch.tensor(0.0).to(0)
    send = torch.tensor(0.0).to(0)

    if rank == 0:

        time1 = 0.0
        for j in range(200):
            print(j)
            input1 = torch.rand(64, 3, 224, 224)
            grad1 = torch.rand(64, 16, 28, 28)
            input1 = input1.chunk(chunk)
            grad1 = grad1.chunk(chunk)
            output1 = []
            input3 = torch.rand(64, 100, 7, 7)
            input3 = input3.chunk(chunk)
            output3 = []
            torch.cuda.synchronize(0)
            start = time.time()
            for i in range(chunk):

                output = layer1(input1[i])
                output1.append(output)
                dist.isend(send, 1)

            for i in range(chunk):
                dist.recv(recv, 1)
                output = layer1(input1[i])
                output = output.sum()
                output3.append(output)
            for i in range(chunk):
                output3[i].backward()
                dist.isend(send, 1)
            for i in range(chunk):
                dist.recv(recv, 1)
                output1[i].backward(grad1[i])
            torch.cuda.synchronize(0)
            end = time.time()
            time1 += end - start
            print(end - start)
        dist.barrier()
        print(time1 / 200)

    else:
        layer2 = layer2.to(1)

        for j in range(200):
            input2 = torch.rand([64, 16, 28, 28]).to(1)
            grad2 = torch.rand([64, 100, 7, 7]).to(1)
            input2 = input2.chunk(chunk)
            grad2 = grad2.chunk(chunk)
            output2 = []
            recv = torch.tensor(0.0).to(1)
            for i in range(chunk):
                dist.recv(recv, 0)
                output = layer2(input2[i])
                dist.isend(recv, 0)
                output2.append(output)
            for i in range(chunk):
                dist.recv(recv, 0)
                output2[i].backward(grad2[i])
                dist.isend(recv, 0)
        dist.barrier()


if __name__ == "__main__":
    main()
