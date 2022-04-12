'''
Author: your name
Date: 2022-04-12 16:29:18
LastEditTime: 2022-04-12 16:41:05
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/test_vision_dgpipe.py
'''
from dist_gpipe_gloo import dist_gpipe,Reshape1
from torchvision.models import mobilenet_v2
import torch.nn as nn
import argparse
import torch
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--chunks", default=4, type=int)
parser.add_argument("--log", default="./log/cv/quant12.txt", type=str)
parser.add_argument("--train-method", default="finetune", type = str)
# parser.add_argument("--warmup", default=0, action="store_true")
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--wd", default=0.0, type=float)
parser.add_argument("--epochs", default=80, type=int)
parser.add_argument("--batches", default=240, type=int)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prune", default=0.0, type=float)
parser.add_argument("--world-size", default=4, type=int)
parser.add_argument("--showperiod", default=30, type=int)
parser.add_argument("--tasktype", default="cv", type=str)
parser.add_argument("--root", default="./data", type=str)
parser.add_argument("--devices", default=[0,1,2,3], type=list)
parser.add_argument("--url", default="tcp://127.0.0.1:1224", type=str)
parser.add_argument("--bachend", default="nccl", type=str)
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--sortquant", default=0, action="store_true")
def main():
    args = parser.parse_args()
    model = mobilenet_v2(pretrained=True)
    model.classifier[-1] = nn.Linear(1280,10)
    devices = [0,1,2,3]
    layer1 = [model.features[0]]
    layer2 = [model.features[1:3]]
    layer3 = [model.features[3:7]]
    layer4 = [model.features[7:]]
    layer5 = [Reshape1(), model.classifier]

    layer1 = nn.Sequential(*layer1)
    layer2 = nn.Sequential(*layer2)
    layer3 = nn.Sequential(*layer3)
    layer4 = nn.Sequential(*layer4)
    layer5 = nn.Sequential(*layer5)
    # input = torch.rand([1,3,224,224])
    # output = layer1(input)
    # print(output.shape)
    # output = layer2(output)
    # print(output.shape)
    # output = layer3(output)
    # print(output.shape)
    # output = layer4(output)
    # print(output.shape)
    # output = layer5(output)
    partition = [[layer1,layer5],[layer2],[layer3],[layer4]]
    tensor_size = [[(int(args.batches/args.chunks),32,112,112),(int(args.batches/args.chunks),1280,7,7)],[(int(args.batches/args.chunks),24,56,56),(int(args.batches/args.chunks),32,112,112)],[(int(args.batches/args.chunks),32,28,28),(int(args.batches/args.chunks),24,56,56)],[(int(args.batches/args.chunks),1280,7,7),(int(args.batches/args.chunks),32,28,28)]]
    print(tensor_size)
    model = dist_gpipe(args,partition,devices,tensor_size)
    model.session()

if __name__ == '__main__':
    main()