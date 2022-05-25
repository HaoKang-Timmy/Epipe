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
from models import *

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--pretrained", default=0, action="store_true")
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epochs", default=15, type=int)
parser.add_argument("--batches", default=70, type=int)
parser.add_argument("--nproc", default=4, type=int)
parser.add_argument("--type", default=0, type=int)
parser.add_argument("--nworker", default=12, type=int)
parser.add_argument("--root", default="../../data", type=str)
parser.add_argument("--savepath", default="./model.pth", type=str)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    print(args.lr)
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    trainset = torchvision.datasets.ImageNet(
        root=args.root, split="train", transform=transform_train
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batches,
        shuffle=(train_sampler is None),
        num_workers=args.nworker,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
    )

    testset = torchvision.datasets.ImageNet(
        root=args.root, split="val", transform=transform_test
    )
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batches,
        shuffle=False,
        num_workers=args.nworker,
        drop_last=True,
        pin_memory=True,
    )
    #     pass
    if args.type == 0:
        model = MobileNetV2withConvInsert()
    elif args.type == 1:
        model = MobileNetV2withConvInsert1_bn()
    elif args.type == 2:
        model = MobileNetV2withConvInsert2()
    elif args.type == 3:
        model = MobileNetV2withConvInsert3_bn()
    device = torch.device("cpu")
    # model.load_state_dict(torch.load(args.savepath+str(args.type),map_location=device))
    model = model.to(rank)

    print(model)
    if args.type != 2:
        optimizer = torch.optim.SGD(
            [
                {"params": model.conv1.parameters()},
                {"params": model.conv2.parameters()},
                {"params": model.t_conv1.parameters()},
                {"params": model.t_conv2.parameters()},
            ],
            lr=args.lr,
            momentum=0.9,
        )
    else:
        optimizer = torch.optim.SGD(
            [
                {"params": model.conv2.parameters()},
                {"params": model.t_conv1.parameters()},
                {"params": model.t_conv3.parameters()},
                {"params": model.conv3.parameters()},
                {"params": model.t_conv2.parameters()},
            ],
            lr=args.lr,
            momentum=0.9,
        )
    model = torch.nn.parallel.DistributedDataParallel(model)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs,
    )
    criterion = nn.CrossEntropyLoss().to(rank)
    best_acc = 0.0
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
        if epoch % 2 == 0 or epoch == 19:
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
            if best_acc < val_acc1:
                best_acc = val_acc1
                if rank == 0:
                    torch.save(
                        model.module.state_dict(), args.savepath + str(args.type)
                    )
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
                    # + str(get_lr(optimizer))
                )
                file_save.close()


if __name__ == "__main__":
    main()
