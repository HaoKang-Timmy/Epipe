from dist_gpipe_client import dist_gpipe_client, Reshape1
from torchvision.models import mobilenet_v2
import torch.nn as nn
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from models.mobilenet_v2_seperate import mobilenet_v2_seperate
from dataloader.cv_dataloader import create_dataloader_cv

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "--chunks", default=8, type=int, help="seperate batchs to chunk minibatches"
)
parser.add_argument(
    "--log",
    default="./test.txt",
    type=str,
    help="name the path and filename of the log file",
)
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--wd", default=0.0, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--batches", default=64, type=int)
parser.add_argument("--loaders", default=12, type=int)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prune", default=0.0, type=float)
parser.add_argument("--world-size", default=2, type=int)
parser.add_argument("--showperiod", default=10, type=int)
parser.add_argument("--tasktype", default="cv", type=str)
parser.add_argument("--root", default="../data", type=str)
parser.add_argument("ranks", default=[0, 1], type=list)
parser.add_argument("rank", default=1, type=int)
parser.add_argument("--local-rank", default=[0], type=list)
parser.add_argument("--ifconfig", default="eno1", type=str)
parser.add_argument("--url", default="tcp://18.25.6.30:23456", type=str)
parser.add_argument("--backend", default="nccl", type=str)
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--sortquant", default=0, action="store_true")
parser.add_argument("--mix", default=0, action="store_true")
parser.add_argument("--pca1", default=0, type=int)
parser.add_argument("--pca2", default=0, type=int)
parser.add_argument("--poweriter1", default=0, type=int)
parser.add_argument("--poweriter2", default=0, type=int)


def main():
    args = parser.parse_args()
    model = mobilenet_v2_seperate(args)
    tensor_size = [
        (int(args.batches / args.chunks), 32, 112, 112),
        (int(args.batches / args.chunks), 1280, 7, 7),
    ]
    train_loader, val_loader = create_dataloader_cv(args)
    model = dist_gpipe_client(args, model, tensor_size, train_loader, val_loader)
