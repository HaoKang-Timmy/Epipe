from numpy import RankWarning
import torchvision.models as models
import torch.nn as nn
import time
from transformers import get_scheduler
import torchvision.transforms as transforms
import torchvision
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from models.models import *
from models.metric import accuracy
from dataloaders.dataloaders import create_dataloaders
from .trainer.trainer import TrainerCV

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--log", default="./test_cifar10_1.txt", type=str)
parser.add_argument("--pretrained", default=0, action="store_true")
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--batches", default=64, type=int)
parser.add_argument("--nproc", default=4, type=int)
parser.add_argument("--type", default=0, type=int)
parser.add_argument("--nworker", default=40, type=int)
parser.add_argument("--root", default="../gpipe_test/data", type=str)
parser.add_argument("--savepath", default="./parameters/mobilenetv2/model", type=str)
parser.add_argument("--dataset", default="CIFAR10", type=str)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=args.nproc, args=(args.nproc, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:1235",
        world_size=args.nproc,
        rank=rank,
    )
    train_loader, val_loader, train_sampler = create_dataloaders(args)
    if args.type == 0:
        model = MobileNetV2withConvInsert0_bn()
    elif args.type == 1:
        model = MobileNetV2withConvInsert1_bn()
    elif args.type == 2:
        model = MobileNetV2withConvInsert2_bn()
    elif args.type == 3:
        model = MobileNetV2withConvInsert3_bn()
    elif args.type == 4:
        model = MobileNetV2withConvInsert4_bn()
    elif args.type == 5:
        model = MobileNetV2withConvInsert5_bn()
    elif args.type == 6:
        model = MobileNetV2withConvInsert6_bn()
    elif args.type == 7:
        model = MobileNetV2withConvInsert7_bn()
    for i, conv in enumerate(model.convsets):
        path = args.savepath + str(args.type) + "conv" + str(i) + ".pth"

        conv.load_state_dict(torch.load(path, map_location="cpu"))

    # model.load_state_dict(torch.load(args.savepath, map_location="cpu"))
    model.mobilenetv2_part3[-1] = torch.nn.Linear(1280, 10)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.SGD(
        [{"params": model.parameters()},], lr=args.lr, momentum=0.9,
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.epochs / 10,
        num_training_steps=args.epochs,
    )
    criterion = nn.CrossEntropyLoss().to(rank)
    best_acc = 0.0
    trainer = TrainerCV(
        model,
        criterion,
        accuracy,
        optimizer,
        train_loader,
        val_loader,
        lr_scheduler,
        rank,
        train_sampler,
        args,
    )
    trainer.traineval()


if __name__ == "__main__":
    main()
