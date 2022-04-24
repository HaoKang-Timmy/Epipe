import time
import torch
import torch.nn as nn
from dataset.dataset_collection import DatasetCollection
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed as dist
from torchgpipe.batchnorm import DeferredBatchNorm
from typing import cast, List


def prepare_dataloader(normalize, compose_train, compose_val, args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    dataset_collection = DatasetCollection(
        args.dataset_type, args.data, compose_train, compose_val
    )
    train_dataset, val_dataset = dataset_collection.init()

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_sampler, train_loader, val_loader


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


def train(model, train_loader, optimizer, criterion, args):
    model.train()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    acc1_avg = 0.0
    loss_avg = 0.0
    start = time.time()
    for i, (images, targets) in enumerate(train_loader):
        model.train()
        images = images.cuda(0, non_blocking=True)
        targets = targets.cuda(0, non_blocking=True)
        output = model(images)
        # print(output.shape,targets.shape)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_time_avg = batch_time_avg + batch_time
        loss_avg = loss_avg + loss
        if i % 30 == 0:
            print("train_loss:", loss, "train_acc1", acc1)
        acc1_avg = acc1 + acc1_avg
        start = time.time()
    batch_time_avg = batch_time_avg / len(train_loader)
    acc1_avg = acc1_avg / len(train_loader)
    loss_avg = loss_avg / len(train_loader)
    print("real", acc1_avg, loss_avg)
    return batch_time_avg, acc1_avg, loss_avg


def validate(model, val_loader, criterion, args):
    model.train()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    acc1_avg = 0.0
    loss_avg = 0.0
    loss_avg_eval = 0.0
    acc1_avg_eval = 0.0
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_time = time.time() - end
            end = time.time()
            batch_time_avg = batch_time_avg + batch_time
            loss_avg = loss_avg + loss
            acc1_avg = acc1_avg + acc1
            if i % 30 == 0:
                print("train_loss:", loss, "train_acc1", acc1)
        batch_time_avg = batch_time_avg / len(val_loader)
        acc1_avg = acc1_avg / len(val_loader)
        loss_avg = loss_avg / len(val_loader)
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_time_avg = batch_time_avg + batch_time
            loss_avg_eval = loss_avg_eval + loss
            acc1_avg_eval = acc1_avg_eval + acc1
            if i % 30 == 0:
                print("train_loss:", loss, "train_acc1", acc1)
        acc1_avg_eval = acc1_avg_eval / len(val_loader)
        loss_avg_eval = loss_avg_eval / len(val_loader)
        return batch_time_avg, acc1_avg, loss_avg, acc1_avg_eval, loss_avg_eval
