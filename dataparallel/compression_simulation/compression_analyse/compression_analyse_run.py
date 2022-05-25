from functions import *
from utils import *
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--fquant", default=6, type=int)
parser.add_argument("--quant", default=8, type=int)
parser.add_argument("--split", default=2, type=int)
parser.add_argument("--pca", default=0, type=int)
args = parser.parse_args()
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
trainset = torchvision.datasets.CIFAR10(
    root=args.root, train=True, download=True, transform=transform_train
)
# quant = torch.rand([64,32,112,112])
with torch.no_grad():
    input = trainset[0][0]

    model = models.mobilenet_v2(pretrained=True)[0]
    input = model(input)
    fastdequant = input.clone().detach()
    stdequant = input.clone().detach()
    min_step = torch.rand([2 ** args.split, 2])
    min_step, fastquant = FastQuantization(input, args.fquant, args.split, min_step)
    fastdequant = FastDequantization(
        fastquant, args.fquant, args.split, min_step, fastdequant
    )
    quant, min, step = QuantizationCPU(input, args.quant)
    dequant = Dequantizationon(quant, args.quant, min, step)

    min_step, stquant = SortQuantization(input, args.fquant, args.split, min_step)
    stdequant = SortDeQuantization(
        stquant, args.fquant, args.split, min_step, stdequant
    )
    abl_q, abl_qmax = abl_difference(input, dequant)
    abl_fq, abl_fqmax = abl_difference(input, fastdequant)
    abl_sq, abl_sqmax = abl_difference(input, stdequant)
    print(abl_q, abl_fq, abl_qmax, abl_fqmax, abl_sq, abl_sqmax)
