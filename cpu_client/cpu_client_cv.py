from dist_gpipe_gloo import dist_gpipe, Reshape1
from torchvision.models import mobilenet_v2
import torch.nn as nn
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--chunks", default=16, type=int)
parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--train-method", default="finetune", type=str)
# parser.add_argument("--warmup", default=0, action="store_true")
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--wd", default=0.0, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--batches", default=64, type=int)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prune", default=0.0, type=float)
parser.add_argument("--world-size", default=2, type=int)
parser.add_argument("--showperiod", default=10, type=int)
parser.add_argument("--tasktype", default="cv", type=str)
parser.add_argument("--root", default="../data", type=str)
parser.add_argument("--devices", default=[0, 1], type=list)
parser.add_argument("--url", default="tcp://127.0.0.1:1226", type=str)
parser.add_argument("--bachend", default="nccl", type=str)
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--sortquant", default=0, action="store_true")
parser.add_argument("--bandwidth", default=0, action="store_true")


def main():
    args = parser.parse_args()
    model = mobilenet_v2(pretrained=True)
    model.classifier[-1] = nn.Linear(1280, 10)
    devices = args.devices
    layer1 = [model.features[0:1]]
    layer2 = [model.features[1:]]
    # layer3 = [model.features[3:7]]
    # layer4 = [model.features[7:]]
    layer5 = [Reshape1(), model.classifier]

    layer1 = nn.Sequential(*layer1)
    layer2 = nn.Sequential(*layer2)
    # layer3 = nn.Sequential(*layer3)
    # layer4 = nn.Sequential(*layer4)
    layer5 = nn.Sequential(*layer5)
    input = torch.rand([1, 3, 224, 224])
    output = layer1(input)
    print(output.shape)
    output = layer2(output)
    print(output.shape)
    # output = layer3(output)
    # print(output.shape)
    # output = layer4(output)
    # print(output.shape)
    output = layer5(output)
    print(output.shape)
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=args.root, train=True, download=True, transform=transform_train
    )
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batches,
        shuffle=True,
        num_workers=12,
        drop_last=True,
        # sampler=train_sampler,
        pin_memory=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.root, train=False, download=True, transform=transform_test
    )
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batches,
        shuffle=False,
        num_workers=12,
        drop_last=True,
        pin_memory=True,
    )
    partition = [[layer1, layer5], [layer2]]
    tensor_size = [
        [
            (int(args.batches / args.chunks), 32, 112, 112),
            (int(args.batches / args.chunks), 1280, 7, 7),
        ],
        # [
        #     (int(args.batches / args.chunks), 24, 56, 56),
        #     (int(args.batches / args.chunks), 32, 112, 112),
        # ],
        # [
        #     (int(args.batches / args.chunks), 32, 28, 28),
        #     (int(args.batches / args.chunks), 24, 56, 56),
        # ],
        [
            (int(args.batches / args.chunks), 1280, 7, 7),
            (int(args.batches / args.chunks), 32, 112, 112),
        ],
    ]
    print(tensor_size)
    model = dist_gpipe(args, partition, devices, tensor_size, train_loader, val_loader)
    model.session()


if __name__ == "__main__":
    main()
