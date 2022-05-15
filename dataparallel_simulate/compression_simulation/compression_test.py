import torchvision.models as models
import torch.nn as nn
import time
from transformers import get_scheduler
import torchvision.transforms as transforms
import torchvision
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from utils import (
    FSQBSVD,
    FSVDBSQ,
    accuracy,
    Reshape1,
    QuantizationLayer,
    DequantizationLayer,
    Fakequantize,
    TopkLayer,
    SortQuantization,
    Fakequantize,
    FQBSQ,
    FastQuantization,
    FSQBQ,
    ChannelwiseQuantization,
    PowerPCA,
    ReshapeSVD,
    PowerSVDLayer,
    PowerSVDLayer1,
)
from powersgd import PowerSGD, Config, optimizer_step


class MobileNetV2withConvInsert1_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert1_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 32, (4, 4), (4, 4))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv2 = nn.Conv2d(1280, 360, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(360, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier
        self.mobilenetv2_part3[-1] = torch.nn.Linear(1280, 10)

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--root", default="../../data", type=str)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=4, args=(4, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:1235", world_size=4, rank=rank
    )
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
        root=args.root, train=True, download=True, transform=transform_train
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=(train_sampler is None),
        num_workers=12,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.root, train=False, download=True, transform=transform_test
    )
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=12,
        drop_last=True,
        pin_memory=True,
    )
    #     pass
    # model = models.mobilenet_v2(pretrained=True)
    # model.classifier[-1] = torch.nn.Linear(1280, 10)
    # layer1 = nn.Sequential(*[model.features[0:1]])
    # feature = model.features[0].children()
    # conv = next(feature)
    # bn = next(feature)

    # layer1 = nn.Sequential(*[conv, bn])
    # layer2 = nn.Sequential(*[nn.ReLU6(inplace=False), model.features[1:]])
    # layer3 = nn.Sequential(*[Reshape1(), model.classifier])

    # quant_layer1 = QuantizationLayer(args.quant)
    # dequant_layer1 = DequantizationLayer(args.quant)
    # quant_layer2 = QuantizationLayer(args.quant)
    # dequant_layer2 = DequantizationLayer(args.quant)
    # layer1 = layer1.to(rank)
    # layer2 = layer2.to(rank)
    # layer3 = layer3.to(rank)
    # conv2d = torch.nn.Conv2d(32, 32, (4, 4), (4, 4)).to(rank)
    # conv_t = torch.nn.ConvTranspose2d(32, 32, (4, 4), (4, 4)).to(rank)
    # conv2d1 = torch.nn.Conv2d(1280, 320, (1, 1)).to(rank)
    # conv_t1 = torch.nn.ConvTranspose2d(320, 1280, (1, 1)).to(rank)
    # quant_layer1 = quant_layer1.to(rank)
    # dequant_layer1 = dequant_layer1.to(rank)
    # quant_layer2 =quant_layer2.to(rank)
    # dequant_layer2 = dequant_layer2.to(rank)
    # layer1 = torch.nn.parallel.DistributedDataParallel(layer1)
    # layer2 = torch.nn.parallel.DistributedDataParallel(layer2)
    # layer3 = torch.nn.parallel.DistributedDataParallel(layer3)
    # conv2d = torch.nn.parallel.DistributedDataParallel(conv2d)
    # conv_t = torch.nn.parallel.DistributedDataParallel(conv_t)
    # conv2d1 = torch.nn.parallel.DistributedDataParallel(conv2d1)
    # conv_t1 = torch.nn.parallel.DistributedDataParallel(conv_t1)
    model = MobileNetV2withConvInsert1_bn()
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model)
    # optimizer = torch.optim.SGD(
    #     [
    #         {"params": layer1.parameters()},
    #         {"params": layer2.parameters()},
    #         {"params": layer3.parameters()},
    #         {"params": conv2d.parameters(), "lr": args.lr},
    #         # {"params": conv2d1.parameters(), "lr": args.lr},
    #         {"params": conv2d1.parameters(), "lr": args.lr},
    #         # {"params": conv2d3.parameters(), "lr": args.lr},
    #         {"params": conv_t.parameters(), "lr": args.lr},
    #         {"params": conv_t1.parameters(), "lr": args.lr},
    #         # {"params": conv_t2.parameters(), "lr": args.lr},
    #         # {"params": conv_t3.parameters(), "lr": args.lr},
    #     ],
    #     lr=args.lr,
    #     momentum=0.9,
    # )
    optimizer = torch.optim.SGD(
        [
            {"params": model.parameters()},
            # {"params": layer2.parameters()},
            # {"params": layer3.parameters()},
            # {"params": conv2d.parameters(), "lr": args.lr},
            # {"params": conv2d1.parameters(), "lr": args.lr},
            # {"params": conv2d1.parameters(), "lr": args.lr},
            # {"params": conv2d3.parameters(), "lr": args.lr},
            # {"params": conv_t.parameters(), "lr": args.lr},
            # {"params": conv_t1.parameters(), "lr": args.lr},
            # {"params": conv_t2.parameters(), "lr": args.lr},
            # {"params": conv_t3.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        momentum=0.9,
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.epochs / 10),
        num_training_steps=args.epochs,
    )

    criterion = nn.CrossEntropyLoss().to(rank)
    bool = 0
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    for epoch in range(args.epochs):
        model.train()

        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        train_acc1 = 0.0
        time_avg = 0.0
        start = time.time()

        for i, (image, label) in enumerate(train_loader):

            image = image.to(rank, non_blocking=True)
            label = label.to(rank, non_blocking=True)

            outputs = model(image)

            # print(outputs)
            # while(1):
            #     pass
            loss = criterion(outputs, label)
            acc, _ = accuracy(outputs, label, topk=(1, 2))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_acc1 += acc.item()

            end = time.time() - start
            time_avg += end
            if i % 20 == 0 and rank == 0:
                print("train_loss", loss.item(), "train_acc", acc.item(), "time", end)
            start = time.time()
        train_loss /= len(train_loader)
        train_acc1 /= len(train_loader)
        time_avg /= len(train_loader)
        lr_scheduler.step()

        model.eval()
        if rank == 0:
            print("lr:", get_lr(optimizer))
        val_loss = 0.0
        val_acc1 = 0.0
        with torch.no_grad():
            for i, (image, label) in enumerate(val_loader):
                image = image.to(rank, non_blocking=True)
                label = label.to(rank, non_blocking=True)

                outputs = model(image)
                loss = criterion(outputs, label)
                acc, _ = accuracy(outputs, label, topk=(1, 2))

                val_loss += loss.item()
                val_acc1 += acc.item()
                if i % 20 == 0 and rank == 0:
                    print("val_loss", loss.item(), "val_acc", acc.item())
            val_loss /= len(val_loader)
            val_acc1 /= len(val_loader)
            print(len(val_loader))
        if rank == 0:
            print(
                "epoch:",
                epoch,
                "train_loss",
                train_loss,
                "train_acc",
                train_acc1,
                "val_loss",
                val_loss,
                "val_acc",
                val_acc1,
            )
            file_save = open(args.log, mode="a")
            file_save.write(
                "\n"
                + "step:"
                + str(epoch)
                + "  loss_train:"
                + str(train_loss)
                + "  acc1_train:"
                + str(train_acc1)
                + "  loss_val:"
                + str(val_loss)
                + "  acc1_val:"
                + str(val_acc1)
                + "  time_per_batch:"
                + str(time_avg)
                + "  lr:"
                + str(get_lr(optimizer))
            )
            file_save.close()


if __name__ == "__main__":
    main()
