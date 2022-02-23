import argparse
from numpy import RankWarning
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils_gpipe import *
from distributed_layers import Reshape
import pytorch_warmup as warmup
from ..models import *
import torchvision
from torchgpipe.batchnorm import DeferredBatchNorm
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1224', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--lr', '--learning-rate', default=0.4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-type', '--dataset-type', default='Imagenet',
                    help='choose a dataset to train')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
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
parser.add_argument('--chunks',default = 4, type = int)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
train_batch_time_sum = 0.0
train_data_time_sum = 0.0
train_acc1_sum = 0.0
val_batch_time_sum = 0.0
val_data_time_sum = 0.0
val_acc1_avg = 0.0
def main_worker(rank,world_size,args):
    global train_batch_time_sum
    global train_data_time_sum
    global train_acc1_sum
    print(args.dist_backend,args.dist_url,args.world_size,rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=rank)
    print("rank:",rank,"init finish")
    torch.cuda.set_device(rank)
    # cudnn.benchmark = True
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    # compose_train = transforms.Compose([
    #     transforms.RandomCrop(32),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # compose_val = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # train_sampler, train_loader, val_loader = prepare_dataloader(
    #     normalize, compose_train, compose_val, args)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=12,drop_last = True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=2*args.batch_size, shuffle=False, num_workers=12,drop_last = True)
    
    if rank == 0:
        
        # model = models.MobileNetV2(num_classes= 10).features[0:4]
        model = MobileNetV2()
        model1 = nn.Sequential(model.conv1,model.bn1)
        model2 = nn.Sequential(model.linear)
        DeferredBatchNorm.convert_deferred_batch_norm(model1,args.chunks)
        DeferredBatchNorm.convert_deferred_batch_norm(model2,args.chunks)
        model1.cuda(rank)
        model2.cuda(rank)
        optimizer = torch.optim.SGD([{'params':model1.parameters()},{'params':model2.parameters()}], args.lr, weight_decay= args.weight_decay,momentum=args.momentum)
        criterion = nn.CrossEntropyLoss().cuda(rank)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        warmup_schduler =warmup.LinearWarmup(optimizer,warmup_period= 20)
        for epoch in range(args.epochs):
            batch_time1, data_time1,acc1_avg,loss_avg = train_header(rank,model1,model2, optimizer,train_loader,criterion,args)
            train_batch_time_sum = train_batch_time_sum + batch_time1
            train_data_time_sum = train_data_time_sum + data_time1
            train_batch_time_avg = train_batch_time_sum / (epoch+1)
            train_data_time_avg = train_data_time_sum / (epoch+1)

            batch_time1, val_acc1, val_loss = val_header(rank,model1,model2,val_loader,criterion,args)
            scheduler.step(scheduler.last_epoch+1)
            warmup_schduler.dampen()
            save_path = "./log/dist_gpipe_test.txt"
            file_save1 = open(save_path, mode='a')
            file_save1.write('\n'+'step:'+str(epoch)+'  loss_train:'+str(loss_avg.item())+'  acc1_train:'+str(
                acc1_avg.item())+'  loss_val:'+str(val_loss.item())+'  acc1_val:'+str(val_acc1.item())+'  time_per_batch:'+str(train_batch_time_avg)+'  time_load_perbatch:'+str(train_data_time_avg))
            print(get_lr(optimizer))
            file_save1.close()
            print("epoch",epoch,"train_batch_time_avg",train_batch_time_avg,"train_data_time_avg",train_data_time_avg,"train_acc1",acc1_avg,"train_loss",loss_avg,"val_acc",val_acc1,"val_loss",val_loss)
            

    elif rank < args.world_size -1:
        if rank == 1:
            model = MobileNetV2(num_classes= 10).layers[0:2]
        else:
            model = MobileNetV2(num_classes= 10).layers[2:7]
        DeferredBatchNorm.convert_deferred_batch_norm(model,args.chunks)
        model.cuda(rank)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay= args.weight_decay,momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        warmup_schduler =warmup.LinearWarmup(optimizer,warmup_period= 10)
        for epoch in range(args.epochs):
            batch_time, data_time = train_medium(rank,model,optimizer,len(train_loader),args)
            train_batch_time_sum = train_batch_time_sum + batch_time
            train_data_time_sum = train_data_time_sum + data_time

            batch_time = val_medium(rank,model,len(val_loader),args)
            scheduler.step(scheduler.last_epoch+1)
            warmup_schduler.dampen()
    elif rank == args.world_size-1:
        model = MobileNetV2()
        model = nn.Sequential(model.layers[7:],model.conv2,model.bn2,Reshape1(),model.linear)
        DeferredBatchNorm.convert_deferred_batch_norm(model,args.chunks)
        model.cuda(rank)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay= args.weight_decay,momentum=args.momentum)
        criterion = nn.CrossEntropyLoss().cuda(rank)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        warmup_schduler =warmup.LinearWarmup(optimizer,warmup_period= 10)
        for epoch in range(args.epochs):
            batch_time3, data_time3 = train_last(rank,model,optimizer,len(train_loader),args)
            train_batch_time_sum = train_batch_time_sum + batch_time3
            train_data_time_sum = train_data_time_sum + data_time3

            batch_time3 = val_last(rank,model,len(val_loader),args)
            scheduler.step(scheduler.last_epoch+1)
            warmup_schduler.dampen()
 

def main():
    args = parser.parse_args()
    mp.spawn(main_worker,nprocs= args.world_size,args=(args.world_size,args))
    pass


if __name__ == '__main__':
    main()