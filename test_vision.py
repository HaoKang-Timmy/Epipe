# from model import resnet
import torchvision.models as models
from dist_gpipe import dist_gpipe,Reshape1,Reshape2
# from model import MobileNetV2, ResNet18
import model as md
import torch.nn as nn
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from dist_gpipe.distributedlayers.distributed_layers import ForwardSendLayers,ForwardReceiveLayers
import torchvision
import torch.distributed as dist
import argparse
# def foo(rank):
#     print("rank",rank)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--chunks',default = 4, type = int)
parser.add_argument('--log-dir',default = './my_gpipe', type = str)
parser.add_argument('--pretrained',default = 0,action= 'store_true')
parser.add_argument('--warmup',default = 0,action= 'store_true')
parser.add_argument('--lr',default = 0.1, type = float)
parser.add_argument('--epoches',default = 100, type = int)
parser.add_argument('--batches',default = 256, type = int)
parser.add_argument('--quant',default = 0, type = int)
parser.add_argument('--prun',default = 0.0, type = float)
parser.add_argument('--Adam',default = 0,action= 'store_true')
def main():
    args = parser.parse_args()
    if args.pretrained:
        model = models.mobilenet_v2(pretrained=True)
        print("use pretrained model")
        model.classifier[-1] = torch.nn.Linear(1280,10)
        layer1 = [model.features[0]]
        layer2 = [model.features[1:3]]
        layer3 = [model.features[3:7]]
        layer4 = [model.features[7:]]
        layer5 = [Reshape1(),model.classifier]
    else:
        model = md.MobileNetV2()
        layer1 = [model.conv1,model.bn1]
        layer2 = [model.layers[0:3]]
        layer3 = [model.layers[3:7]]
        layer4 = [model.layers[7:],model.conv2]
        layer5 = [model.bn2,Reshape1(), model.linear]
    
        print("use kuangliu model")
    print(model)



    # input = torch.rand([512,3,32,32])
    # input = layer1(input)
    # input = layer2(input)
    # input = layer3(input)
    # input = layer4(input)
    # input = layer5(input)
    layer1 = nn.Sequential(*layer1)
    layer2 = nn.Sequential(*layer2)
    layer3 = nn.Sequential(*layer3)
    layer4 = nn.Sequential(*layer4)
    layer5 = nn.Sequential(*layer5)
    # input = torch.rand([4,3,32,32])
    # input = layer1(input)
    # input = layer2(input)
    # input = layer3(input)
    # input = layer4(input)
    # output = layer5(input)
    # print(output.shape)
    partition = [layer1,layer2,layer3,layer4,layer5]
    criterion = nn.CrossEntropyLoss()
    # transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    if args.pretrained:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        input_size = [args.batches,3,224,224]
    else:
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
        input_size = [args.batches,3,32,32]
    trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batches, shuffle=True, num_workers=12,drop_last = True)

    testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.batches, shuffle=False, num_workers=12,drop_last = True)
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
    settings = {}
    settings['lr'] = args.lr
    settings['wd'] = 1e-4
    settings['momentum'] = 0.9
    settings['quantization'] = args.quant
    print(args.warmup)
    settings['warmup'] = args.warmup
    # if args.prun != 0.0:
    settings['prun'] = args.prun
    # else:
    #     settings['prun'] = None
    settings['Adam'] = args.Adam
    model  = dist_gpipe(partition,[0,1,2,3,0],args.chunks,input_size=input_size,criterion = criterion,save_path = args.log_dir,settings = settings)

    # len_train_loader,len_val_loader = len(train_loader),len(val_loader)
    
    model.session(train_loader,val_loader,args.epoches,settings)
    
if __name__ == '__main__':
    main()

