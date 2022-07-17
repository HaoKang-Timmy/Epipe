"""
Author: Beta Cat 466904389@qq.com
Date: 2022-07-14 20:38:58
LastEditors: Beta Cat 466904389@qq.com
LastEditTime: 2022-07-18 02:50:30
FilePath: /research/gpipe_test/dataparallel/compression_simulation/dataparallel_test_nlp.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import torch.nn as nn
import time
import torch
from datasets import load_dataset
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_metric
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import os
from trainer.trainer import TrainerNLP
from torch.optim import Adam
from dataloader.dataloader import create_dataloader_nlp
from models.models import Robertawithcompress


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--log", default="./test_hg.txt", type=str)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--wd", default=0.0001, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--task", default="cola", type=str)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prun", default=0.0, type=float)
parser.add_argument("--batches", default=32, type=int)
parser.add_argument("--sort", default=0, type=int)
parser.add_argument("--pca", default=0, type=int)
parser.add_argument("--powerpca", default=0, type=int)
parser.add_argument("--poweriter", default=2, type=int)
parser.add_argument("--linear", default=0, type=int)
parser.add_argument("--nproc", default=4, type=int)
parser.add_argument("--nworker", default=12, type=int)
parser.add_argument("--sortquant", default=0, action="store_true")
parser.add_argument("--mix", default=0, action="store_true")


def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=args.nproc, args=(args.nproc, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:1237", world_size=4, rank=rank
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train_dataloader, val_dataloader, train_sampler = create_dataloader_nlp(args)
    # metric
    metric_mat = load_metric("glue", args.task)
    metric_acc = load_metric("accuracy")
    # model
    epochs = args.epochs

    model = Robertawithcompress(args, rank, args.batches)
    model = model.to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        eps=1e-06,
        betas=(0.9, 0.98),
    )

    lr_scheduler = get_scheduler(
        name="polynomial",
        optimizer=optimizer,
        num_warmup_steps=320,
        num_training_steps=epochs * len(train_dataloader),
    )
    print(len(train_dataloader))
    print(len(val_dataloader))
    criterion = nn.CrossEntropyLoss().to(rank)
    trainer = TrainerNLP(
        model,
        criterion,
        metric_mat,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        rank,
        train_sampler,
        args,
        metric_acc,
    )
    trainer.traineval()


if __name__ == "__main__":
    main()
