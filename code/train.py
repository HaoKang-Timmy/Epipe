import argparse
from numpy import RankWarning
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils import *
from distributed_layers import Reshape,Topk,TopkAbs,QuantLayer,DeQuantLayer
import pytorch_warmup as warmup
from models import *
import torchvision
from torchgpipe import GPipe
from typing import cast,List
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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
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
    trainset, batch_size=1024, shuffle=False, num_workers=12,drop_last = True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(
    testset, batch_size=2048, shuffle=False, num_workers=12,drop_last = True)
model = MobileNetV2()
partition = []
#print(model)
# model1 = [model.conv1,model.bn1,model.layers[0:3]]
# model2 = [model.layers[3:6]]
# model3 = [model.layers[6:9]]
# model4 = [model.layers[9:12]]
# model5 = [model.layers[12:15]]
# model6 = [model.layers[15:],model.conv2,model.bn2,Reshape1(),model.linear]
# print(model1)
# model1 = nn.Sequential(*model1,*model2,*model3,*model4,*model5,*model6)
# print(model1[5])
# model = nn.Sequential([model.conv1,model.bn1,model.layers[0:3]],model.layers[3:9],model.layers[9:15],[model.layers[15:],model.conv2,model.bn2,Reshape1(),model.linear])

layer1 = nn.Sequential(model.conv1,model.bn1,QuantLayer(16))
DeferredBatchNorm.convert_deferred_batch_norm(layer1,4)
layer1 = layer1.to(0)
layer2 = nn.Sequential(DeQuantLayer(16),model.layers[0:7])
DeferredBatchNorm.convert_deferred_batch_norm(layer2,4)
layer2 = layer2.to(1)
layer3 = nn.Sequential(model.layers[7:14])
DeferredBatchNorm.convert_deferred_batch_norm(layer3,4)
layer3 =layer3.to(2)
layer4 = nn.Sequential(model.layers[14:],QuantLayer(16)).to(3)
DeferredBatchNorm.convert_deferred_batch_norm(layer4,4)
layer4 = layer4.to(3)
layer5 = nn.Sequential(DeQuantLayer(16),model.conv2,model.bn2,Reshape1(), model.linear)
DeferredBatchNorm.convert_deferred_batch_norm(layer5,4)
layer5 =layer5.to(0)


# layer1 = nn.Sequential(model.conv1,model.bn1)
# DeferredBatchNorm.convert_deferred_batch_norm(layer1,4)
# layer1 = layer1.to(0)
# layer2 = nn.Sequential(model.layers[0:7])
# DeferredBatchNorm.convert_deferred_batch_norm(layer2,4)
# layer2 = layer2.to(1)
# layer3 = nn.Sequential(model.layers[7:14])
# DeferredBatchNorm.convert_deferred_batch_norm(layer3,4)
# layer3 =layer3.to(2)
# layer4 = nn.Sequential(model.layers[14:]).to(3)
# DeferredBatchNorm.convert_deferred_batch_norm(layer4,4)
# layer4 = layer4.to(3)
# layer5 = nn.Sequential(model.conv2,model.bn2,Reshape1(), model.linear)
# DeferredBatchNorm.convert_deferred_batch_norm(layer5,4)
# layer5 =layer5.to(0)

partition.append(layer1)
partition.append(layer2)
partition.append(layer3)
partition.append(layer4)
partition.append(layer5)
partition = cast(List[nn.Sequential], nn.ModuleList(partition))
model = GPipe(model,partition = partition,devices = [0,1,2,3,0], chunks=4)
# model = model.to(0)
args = parser.parse_args()
optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay= args.weight_decay,momentum=args.momentum)
criterion = nn.CrossEntropyLoss().cuda(3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
warmup_schduler =warmup.LinearWarmup(optimizer,warmup_period= 20)
# input = torch.rand([128,3,32,32]).to(0)
# output = model(input)
# print(output.get_device())
for epoch in range(args.epochs):
    train_batch_time, acc1_train,loss_train =train(model,train_loader,optimizer,criterion,args)
    scheduler.step(scheduler.last_epoch+1)
    warmup_schduler.dampen()
    val_batch_time, val_acc, loss_val = validate(model,val_loader,criterion,args)
    train_loss_save = './log/gpipe_prune0.1_1024_4chunks.txt'
    file_save1 = open(train_loss_save, mode='a')
    file_save1.write('\n'+'step:'+str(epoch)+'  loss_train:'+str(loss_train.item())+'  acc1_train:'+str(
        acc1_train.item())+'  loss_val:'+str(loss_val.item())+'  acc1_val:'+str(val_acc.item())+'  time_per_batch:'+str(train_batch_time))
    file_save1.close()





