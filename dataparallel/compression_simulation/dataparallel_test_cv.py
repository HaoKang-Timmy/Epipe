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
from utils import (
    FSQBSVD,
    FSVDBSQ,
    accuracy,
    Reshape1,
    QuantizationLayer,
    DequantizationLayer,
    Fakequantize,
    TopkLayer,
    SortQuantization,
    Fakequantize,
    FQBSQ,
    FastQuantization,
    FSQBQ,
    ChannelwiseQuantization,
    PowerPCA,
    ReshapeSVD,
    PowerSVDLayer,
    PowerSVDLayer1,
)
from powersgd import PowerSGD, Config, optimizer_step

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--pretrained", default=0, action="store_true")
parser.add_argument("--warmup", default=0, action="store_true")
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--batches", default=64, type=int)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prune", default=0.0, type=float)
parser.add_argument("--avgpool", default=0, action="store_true")
parser.add_argument("--secondlayer", default=0, action="store_true")
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--sortquant", default=0, action="store_true")
parser.add_argument("--qsq", default=0, action="store_true")
parser.add_argument("--svdq", default=0, action="store_true")
parser.add_argument("--kmeans", default=0, type=int)
parser.add_argument("--qquant", default=0, type=int)
parser.add_argument("--channelquant", default=0, type=int)
parser.add_argument("--pca1", default=0, type=int)
parser.add_argument("--pca2", default=0, type=int)
parser.add_argument("--root", default="../../data", type=str)
parser.add_argument("--conv1", default=0, action="store_true")
parser.add_argument("--conv2", default=0, action="store_true")
parser.add_argument("--conv1kernel", default=0, type=tuple)
parser.add_argument("--powersvd", default=0, type=int)
parser.add_argument("--powersvd1", default=0, type=int)
parser.add_argument("--poweriter", default=2, type=int)
parser.add_argument("--svd", default=0, type=int)
parser.add_argument("--loader", default=12, type=int)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=4, args=(4, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:1235", world_size=4, rank=rank
    )
    train_loader, val_loader,train_sampler = create_dataloader_cv(args)
    #     pass
    model = MobileNetV2Compress(args,rank,[args.batches,32,112,112],[args.batches,1280,7,7])

    model = model.to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model)
 
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.epochs / 10),
        num_training_steps=args.epochs,
    )

    criterion = nn.CrossEntropyLoss().to(rank)
    bool = 0
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        train_acc1 = 0.0
        time_avg = 0.0
        start = time.time()

        for i, (image, label) in enumerate(train_loader):

            image = image.to(rank, non_blocking=True)
            label = label.to(rank, non_blocking=True)
            outputs = model(image)
            
            loss = criterion(outputs, label)
            acc, _ = accuracy(outputs, label, topk=(1, 2))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_acc1 += acc.item()

            end = time.time() - start
            time_avg += end
            if i % 20 == 0 and rank == 0:
                print("train_loss", loss.item(), "train_acc", acc.item(), "time", end)
            start = time.time()
        train_loss /= len(train_loader)
        train_acc1 /= len(train_loader)
        time_avg /= len(train_loader)
        lr_scheduler.step()

        model.eval()
        if rank == 0:
            print("lr:", get_lr(optimizer))
        val_loss = 0.0
        val_acc1 = 0.0
        with torch.no_grad():
            for i, (image, label) in enumerate(val_loader):
                image = image.to(rank, non_blocking=True)
                label = label.to(rank, non_blocking=True)

                outputs = model(image)
                loss = criterion(outputs, label)
                acc, _ = accuracy(outputs, label, topk=(1, 2))

                val_loss += loss.item()
                val_acc1 += acc.item()
                if i % 20 == 0 and rank == 0:
                    print("val_loss", loss.item(), "val_acc", acc.item())
            val_loss /= len(val_loader)
            val_acc1 /= len(val_loader)
            print(len(val_loader))
        if rank == 0:
            print(
                "epoch:",
                epoch,
                "train_loss",
                train_loss,
                "train_acc",
                train_acc1,
                "val_loss",
                val_loss,
                "val_acc",
                val_acc1,
            )
            file_save = open(args.log, mode="a")
            file_save.write(
                "\n"
                + "step:"
                + str(epoch)
                + "  loss_train:"
                + str(train_loss)
                + "  acc1_train:"
                + str(train_acc1)
                + "  loss_val:"
                + str(val_loss)
                + "  acc1_val:"
                + str(val_acc1)
                + "  time_per_batch:"
                + str(time_avg)
                + "  lr:"
                + str(get_lr(optimizer))
            )
            file_save.close()


if __name__ == "__main__":
    main()
