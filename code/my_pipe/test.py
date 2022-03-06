'''
Author: your name
Date: 2022-02-27 17:32:46
LastEditTime: 2022-03-05 18:06:03
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/code/my_pipe/test.py
'''
from my_pipe import dist_gpipe,Reshape1
from models import MobileNetV2
import torch.nn as nn
import torch
import multiprocessing as mp
import torchvision.transforms as transforms
from my_pipe.distributedlayers.distributed_layers import ForwardSendLayers,ForwardReceiveLayers
import torchvision
import torch.distributed as dist
import argparse
# def foo(rank):
#     print("rank",rank)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--chunks',default = 1, type = int)
def main():
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    model = MobileNetV2()
    layer1 = [model.conv1,model.bn1]
    layer2 = [model.layers[0:3]]
    layer3 = [model.layers[3:7]]
    layer4 = [model.layers[7:],model.conv2]
    layer5 = [model.bn2,Reshape1(), model.linear]
    layer1 = nn.Sequential(*layer1)
    layer2 = nn.Sequential(*layer2)
    layer3 = nn.Sequential(*layer3)
    layer4 = nn.Sequential(*layer4)
    layer5 = nn.Sequential(*layer5)
    partition = [layer1,layer2,layer3,layer4,layer5]
    criterion = nn.CrossEntropyLoss()
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
    trainset, batch_size=512, shuffle=True, num_workers=12,drop_last = True)

    testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=False, num_workers=12,drop_last = True)
#     for i ,(images,targets) in enumerate(train_loader):
#         print(targets.shape)
    # print(torch.tensor([1,2,3,4]).shape)
    # for i, (images,targets) in enumerate(train_loader):
    #     if i ==0:
                
    #             model = dist_gpipe(partition,[0,1,2,3,0],4,input_size=list(images.shape),criterion = criterion)
    #             train_acc,train_loss = model.session(input = images,targets=targets)
    #             break
    # model = dist_gpipe(partition,[0,1,2,3,0],2,input_size=[4,3,32,32],criterion = criterion)
    # train_acc,train_loss = model.train(input = torch.rand(4,3,32,32),targets=torch.tensor([1,2,3,1]))
    model  = dist_gpipe(partition,[0,1,2,3,0],args.chunks,input_size=[512,3,32,32],criterion = criterion,save_path = "./mygpipe_log/mobilenet/512b_0.2lr_origin_4chunks.txt")
    settings = {}
    settings['lr'] = 0.2
    settings['wd'] = 1e-4
    settings['momentum'] = 0.9
    len_train_loader,len_val_loader = len(train_loader),len(val_loader)
    del train_loader, val_loader
    model.session(len_train_loader,len_val_loader,100,settings)
    
if __name__ == '__main__':
    main()

