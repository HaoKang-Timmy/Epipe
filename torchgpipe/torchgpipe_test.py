import argparse
from numpy import RankWarning
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils import *

#
import pytorch_warmup as warmup
from model import *
import torchvision
from torchgpipe import GPipe
from typing import cast, List
from torchgpipe.batchnorm import DeferredBatchNorm


def getlr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--root", type=str, default="../data")
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:1224",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.4,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-type", "--dataset-type", default="Imagenet", help="choose a dataset to train"
)
parser.add_argument(
    "-b",
    "--batches",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "-j",
    "--workers",
    default=12,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 12)",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--train-type",
    default="cloud+pruning+quantization",
    type=str,
    help="train and pruning method",
)
parser.add_argument("--pruning", default=0.1, type=float)
parser.add_argument("--scalar", default=16, type=float)
parser.add_argument("-logdir", type=str)


class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        return out


args = parser.parse_args()

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
model = MobileNetV2()
model.classifier[-1] = nn.Linear(1280, 10)

layer1 = [model.features[0:1]]
layer2 = [model.features[1:0]]
layer3 = [Reshape1(), model.classifier]
layer1 = nn.Sequential(*layer1).to(0)
layer2 = nn.Sequential(*layer2).to(1)
layer3 = nn.Sequential(*layer3).to(0)
partition = [layer1, layer2, layer3]
partition = cast(List[nn.Sequential], nn.ModuleList(partition))
model = GPipe(model, partition=partition, devices=[0, 1, 0], chunks=4)
optimizer = torch.optim.SGD(
    model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum
)
criterion = nn.CrossEntropyLoss().cuda(3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
warmup_schduler = warmup.LinearWarmup(optimizer, warmup_period=20)
# input = torch.rand([128,3,32,32]).to(0)
# output = model(input)
# print(output.get_device())
for epoch in range(args.epochs):
    print(epoch)
    train_batch_time, acc1_train, loss_train = train(
        model, train_loader, optimizer, criterion, args
    )
    scheduler.step(scheduler.last_epoch + 1)
    warmup_schduler.dampen()
    # print("lr",scheduler.get_lr())
    # print("last_r",scheduler.get_last_lr())
    # print("opt",getlr(optimizer))
    val_batch_time, val_acc, loss_val, acc_val_eval, loss_val_eval = validate(
        model, val_loader, criterion, args
    )
    train_loss_save = args.logdir
    file_save1 = open(train_loss_save, mode="a")
    file_save1.write(
        "\n"
        + "step:"
        + str(epoch)
        + "  loss_train:"
        + str(loss_train.item())
        + "  acc1_train:"
        + str(acc1_train.item())
        + "  loss_val:"
        + str(loss_val.item())
        + "  acc1_val:"
        + str(val_acc.item())
        + "  time_per_batch:"
        + str(train_batch_time)
        + "  loss_val_eval:"
        + str(loss_val_eval.item())
        + "  acc_val_eval:"
        + str(acc_val_eval.item())
        + "  lr:"
        + str(scheduler.get_lr()[0])
    )
    file_save1.close()
