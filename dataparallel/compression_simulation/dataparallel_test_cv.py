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
from models.models import MobileNetV2Compress
from dataloader.dataloader import create_dataloader_cv
from logger.config import create_config
from utils import accuracy
from trainer.trainer import TrainerCV

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--batches", default=64, type=int)
parser.add_argument("--showperiod", default=30, type=int)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prune", default=0.0, type=float)
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--c", default="./", type=str)
parser.add_argument("--sortquant", default=0, action="store_true")
parser.add_argument("--secondlayer", default=0, action="store_true")
parser.add_argument("--root", default="../../data", type=str)
parser.add_argument("--conv1", default=0, action="store_true")
parser.add_argument("--conv2", default=0, action="store_true")
parser.add_argument("--conv1kernel", default=0, type=tuple)
parser.add_argument("--powerrank", default=0, type=int)
parser.add_argument("--powerrank1", default=0, type=int)
parser.add_argument("--poweriter", default=2, type=int)
parser.add_argument("--svd", default=0, type=int)
parser.add_argument("--loader", default=12, type=int)
parser.add_argument("--worker", default=4, type=int)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def main():
    args = parser.parse_args()
    config = create_config(args)
    f = open((args.c + "exp.yaml"), "a")
    f.write(config.__str__())
    mp.spawn(main_worker, nprocs=args.worker, args=(args.worker, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:1235",
        world_size=args.worker,
        rank=rank,
    )
    train_loader, val_loader, train_sampler = create_dataloader_cv(args)
    #     pass
    model = MobileNetV2Compress(
        args, rank, [args.batches, 32, 112, 112], [args.batches, 1280, 7, 7]
    )

    model = model.to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.epochs / 10),
        num_training_steps=args.epochs,
    )

    criterion = nn.CrossEntropyLoss().to(rank)

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
