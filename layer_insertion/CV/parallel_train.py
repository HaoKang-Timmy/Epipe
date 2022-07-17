"""
Author: Beta Cat 466904389@qq.com
Date: 2022-07-14 20:38:58
LastEditors: Beta Cat 466904389@qq.com
LastEditTime: 2022-07-16 22:40:36
FilePath: /research/gpipe_test/layer_insertion/CV/parallel_train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
from venv import create
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
from models.optimizer import create_optimizer
from models.metric import accuracy
from dataloaders.dataloaders import create_dataloaders
from trainer.trainer import TrainerCV

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--pretrained", default=0, action="store_true")
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batches", default=120, type=int)
parser.add_argument("--nproc", default=4, type=int)
parser.add_argument("--type", default=0, type=int)
parser.add_argument("--nworker", default=40, type=int)
parser.add_argument("--root", default="/dataset/imagenet", type=str)
parser.add_argument("--savepath", default="./model", type=str)
parser.add_argument("--dataset", default="IMAGENET", type=str)


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
    args.lr = args.batches * process_num / 256 * args.lr
    train_loader, val_loader, train_sampler = create_dataloaders(args)
    if args.type == 0:
        model = MobileNetV2withConvInsert0_bn()
    elif args.type == 1:
        model = MobileNetV2withConvInsert1_bn()
    elif args.type == 2:
        model = MobileNetV2withConvInsert2_bn()
    elif args.type == 3:
        model = MobileNetV2withConvInsert3_bn()
    # double insertion
    elif args.type == 4:
        model = MobileNetV2withConvInsert4_bn()
    elif args.type == 5:
        model = MobileNetV2withConvInsert5_bn()
    elif args.type == 6:
        model = MobileNetV2withConvInsert6_bn()
    elif args.type == 7:
        model = MobileNetV2withConvInsert7_bn()
    model = model.to(rank)
    optimizer = create_optimizer(args, model)

    model = torch.nn.parallel.DistributedDataParallel(model)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=2,
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
    trainer.train()


if __name__ == "__main__":
    main()
