"""
Author: Beta Cat 466904389@qq.com
Date: 2022-07-14 20:38:58
LastEditors: Beta Cat 466904389@qq.com
LastEditTime: 2022-07-18 02:56:46
FilePath: /research/gpipe_test/layer_insertion/CV/dataloaders/dataloaders.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import torchvision.transforms as transforms
import torchvision
import torch


def create_dataloaders(args):
    if args.dataset == "IMAGENET":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        trainset = torchvision.datasets.ImageNet(
            root=args.root, split="train", transform=transform_train
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batches,
            shuffle=(train_sampler is None),
            num_workers=args.nworker,
            drop_last=True,
            sampler=train_sampler,
            pin_memory=True,
        )

        testset = torchvision.datasets.ImageNet(
            root=args.root, split="val", transform=transform_test
        )
        val_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batches,
            shuffle=False,
            num_workers=args.nworker,
            drop_last=True,
            pin_memory=True,
        )
    elif args.dataset == "CIFAR10":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=args.root, train=True, download=True, transform=transform_train
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

        testset = torchvision.datasets.CIFAR10(
            root=args.root, train=False, download=True, transform=transform_test
        )
        val_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batches,
            shuffle=False,
            num_workers=args.nworker,
            drop_last=True,
            pin_memory=True,
        )
    elif args.dataset == "CIFAR100":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        trainset = torchvision.datasets.CIFAR100(
            root=args.root, train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR100(
            root=args.root, train=False, download=True, transform=transform_test
        )
    elif args.dataset == "FOOD101":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        trainset = torchvision.datasets.Food101(
            root=args.root, split="train", download=True, transform=transform_train
        )
        testset = torchvision.datasets.Food101(
            root=args.root, split="test", download=True, transform=transform_test
        )
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batches,
        shuffle=(train_sampler is None),
        num_workers=args.nworker,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batches,
        shuffle=False,
        num_workers=args.nworker,
        drop_last=True,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler
