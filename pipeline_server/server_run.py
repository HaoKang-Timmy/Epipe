from dist_gpipe_server import dist_gpipe_server
from torchvision.models import mobilenet_v2
import torch.nn as nn
import argparse
import torch
from models.mobilenetv2_seperate import mobilenet_v2_seperate

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
# parser.add_argument("--warmup", default=0, action="store_true")
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--wd", default=0.0, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--batches", default=64, type=int)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prune", default=0.0, type=float)
parser.add_argument("--world-size", default=2, type=int)
parser.add_argument("--showperiod", default=10, type=int)
parser.add_argument("--tasktype", default="cv", type=str)
parser.add_argument("--root", default="../data", type=str)
parser.add_argument("--ranks", default=[0, 1], type=list)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--local-rank", default=[0], type=list)
parser.add_argument("--ifconfig", default="enp14s0", type=str)
parser.add_argument("--url", default="tcp://18.25.6.30:23456", type=str)
parser.add_argument("--backend", default="nccl", type=str)
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--sortquant", default=0, action="store_true")
parser.add_argument("--mix", default=0, action="store_true")
parser.add_argument("--convinsert", default=0, action="store_true")
parser.add_argument("--pca1", default=0, type=int)
parser.add_argument("--pca2", default=0, type=int)
parser.add_argument("--poweriter1", default=0, type=int)
parser.add_argument("--poweriter2", default=0, type=int)


def main():
    args = parser.parse_args()
    model = mobilenet_v2_seperate(args)
    tensor_size = [
        (int(args.batches / args.chunks), 1280, 7, 7),
        (int(args.batches / args.chunks), 32, 112, 112),
    ]

    # cifar10 settings
    len_trainloader = int(50000 / args.batches)
    len_valloader = int(10000 / args.batches)
    model = dist_gpipe_server(
        args, model, args.local_rank, tensor_size, len_trainloader, len_valloader
    )
    model.session()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
