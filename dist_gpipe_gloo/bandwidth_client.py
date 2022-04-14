import argparse
from numpy import RankWarning
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import models
from torch.optim import AdamW, SGD
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
# from utils import *
# from distributed_layers import Reshape
import pytorch_warmup as warmup
# from model import *
import torchvision
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
from distributedlayers.distributed_nccl_layers import FRBSFunction,FSBRFunction
from torchvision.models import mobilenet_v2
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
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1224', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--world-size', default=2, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--lr', '--learning-rate', default=0.4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-type', '--dataset-type', default='Imagenet',
                    help='choose a dataset to train')
parser.add_argument('-b', '--batches', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--chunk', default=4, type=int)
def main_worker(rank,world_size,args):
    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=rank)
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
        root="../data", train=True, download=True, transform=transform_train
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
        root="../data", train=False, download=True, transform=transform_test
    )
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batches,
        shuffle=False,
        num_workers=12,
        drop_last=True,
        pin_memory=True,
    )
    
    
    if rank == 0:
        # dataloader

        model = mobilenet_v2(pretrained=True)
        model.classifier[-1] = nn.Linear(1280, 10)
        layer1 = [model.features[0]]
        # layer3 = [model.classifier]
        layer3 = nn.Linear(1280, 10)
        layer1 = nn.Sequential(*layer1)
        # layer3= nn.Sequential(*layer3)
        layer1 =layer1.to(0)
        layer3 = layer3.to(0)
        param_list = []
        param_list.append({"params": layer1.parameters()})
        param_list.append({"params": layer3.parameters()})
        optimizer = SGD(
            param_list, lr=0.005, weight_decay=0.0
        )
        criterion = nn.CrossEntropyLoss().to(rank)
        reshape = Reshape1()
        for epoch in range(args.epochs):
            # train
            layer1.train()
            layer3.train()
            acc1_avg, losses_avg = 0.0,0.0
            for batchiter,(image,label) in enumerate(train_loader):
                image = image.to(rank,non_blocking = True)
                label = label.to(rank,non_blocking = True)
                image = image.chunk(args.chunk)
                label = label.chunk(args.chunk)
                batch1 = []
                batch2 = []
                acc1 = 0.0
                losses = 0.0
                for chunk in range(args.chunk):
                    
                    output = layer1(image[chunk])

                    output = FSBRFunction.apply(output,1,0)
                    batch1.append(output)
                for chunk in range(args.chunk):
                    recv = torch.zeros([int(args.batches/args.chunk),1280,7,7]).to(0).requires_grad_()

                    recv = FRBSFunction.apply(recv,1,0)
                    print("before")

                    recv =reshape(recv)

                    output = layer3(recv)
                    # print("after")
                    loss = criterion(output,label[chunk])
                    acc,_ = accuracy(output,label[chunk],topk =(1,2))
                    batch2.append(loss)
                    acc1 += acc.item()
                    losses = loss.item()
                acc1 /= args.chunk
                losses /= args.chunk
                acc1_avg, losses_avg = acc1 + acc1_avg, losses_avg + losses
                if batchiter % 10 == 0:
                    print("tarining_loss:", losses, "training_acc", acc)
                # for back in range(1, -1, -1):
                for chunk in range(args.chunk):
                    batch2[chunk].backward()
                for chunk in range(args.chunk):
                    torch.zeros([])
                    batch1[chunk].backward(torch.zeros([int(args.batches/args.chunk),32,112,112]).to(0))
                optimizer.step()
                optimizer.zero_grad()
            acc1_avg/= len(train_loader)
            losses_avg /=len(train_loader)
            # train
    else:
        model = mobilenet_v2(pretrained=True)
        model.classifier[-1] = nn.Linear(1280, 10)
        
        layer2 = [model.features[1:]]
        layer2 = nn.Sequential(*layer2)
        layer2 = layer2.to(rank)
        optimizer = SGD(
            layer2.parameters(), lr=0.005, weight_decay=0.0
        )
        for epoch in range(args.epochs):
            layer2.train()
            for batch_iter in range(len(train_loader)):
                batch = []
                for chunk in range(args.chunk):
                    input = (
                        torch.zeros((int(args.batches/args.chunk),32,112,112))
                        .to(rank)
                        .requires_grad_()
                    )
                    input = FRBSFunction.apply(input,0,1)
                    output = layer2(input)
                    output = FSBRFunction.apply(output,0,1)
                    batch.append(output)
                for chunk in range(args.chunk):
                    batch[chunk].backward(torch.zeros(tuple(list(batch[chunk].shape))).to(
                            rank
                        ))
                optimizer.step()
                optimizer.zero_grad()


            

def main():
    args = parser.parse_args()
    main_worker(0,2,args)


if __name__ == '__main__':
    main()