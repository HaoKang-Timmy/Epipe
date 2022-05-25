from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_scheduler
from datasets import load_dataset
import argparse
import torch
import os
from datasets import load_metric
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from utils import *
from dataset.dataloader import create_dataloader

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--task", default="rte", type=str)
parser.add_argument("--batches", default=8, type=int)
parser.add_argument("--rank", default=100, type=int)
parser.add_argument("--compressdim", default=-1, type=int)
parser.add_argument("--type", default=3, type=int)


def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=4, args=(4, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:1237", world_size=4, rank=rank
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    train_dataloader, val_dataloader, train_sampler = create_dataloader(args)
    epochs = args.epochs
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    for param in model.parameters():
        param.require_grad = False
    if args.type == 3:
        embedding = model.roberta.embeddings
        attention = model.roberta.encoder.layer[0].attention
        medium = model.roberta.encoder.layer[0].intermediate
        output_layer = model.roberta.encoder.layer[0].output
        roberta_layers = model.roberta.encoder.layer[1:]

        part1 = EmbeddingAndAttention([embedding], [attention])
        part2 = CombineLayer([medium], [output_layer], [roberta_layers])
        part3 = model.classifier
        part1.to(rank)
        part2.to(rank)
        part3.to(rank)
    elif args.type == 4:
        embedding = model.roberta.embeddings
        part1 = model.roberta.encoder.layer[0]
        part2 = NLPSequential([model.roberta.encoder.layer[1:]])
        part3 = model.classifier
        embedding = embedding.to(rank)
        part1.to(rank)
        part2.to(rank)
        part3.to(rank)
    if args.compressdim == -1:
        linear1 = nn.Linear(768, args.rank).to(rank)
        linear2 = nn.Linear(args.rank, 768).to(rank)
        linear3 = nn.Linear(768, args.rank).to(rank)
        linear4 = nn.Linear(args.rank, 768).to(rank)
    elif args.compressdim == -2:
        linear1 = nn.Linear(128, args.rank).to(rank)
        linear2 = nn.Linear(args.rank, 128).to(rank)
        linear3 = nn.Linear(128, args.rank).to(rank)
        linear4 = nn.Linear(args.rank, 128).to(rank)
    linear1 = torch.nn.parallel.DistributedDataParallel(linear1)
    linear2 = torch.nn.parallel.DistributedDataParallel(linear2)
    linear3 = torch.nn.parallel.DistributedDataParallel(linear3)
    linear4 = torch.nn.parallel.DistributedDataParallel(linear4)
    optimizer = AdamW(
        [
            {"params": linear1.parameters()},
            {"params": linear2.parameters()},
            {"params": linear3.parameters()},
            {"params": linear4.parameters()},
        ],
    )
    lr_scheduler = get_scheduler(
        name="polynomial",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=epochs * len(train_dataloader),
    )
    criterion = nn.MSELoss().to(rank)
    print(len(train_dataloader))
    for epoch in range(epochs):
        print(epoch)
        part1.train()
        part2.train()
        part3.train()
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(rank) for k, v in batch.items()}
            batch["attention_mask"] = (
                torch.reshape(
                    batch["attention_mask"],
                    [
                        int(batch["attention_mask"].shape[0]),
                        1,
                        1,
                        int(batch["attention_mask"].shape[-1]),
                    ],
                )
                .to(rank)
                .type(torch.float32)
            )
            batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e9
            if args.type == 4:
                batch["input_ids"] = embedding(batch["input_ids"])
            labels1 = part1(batch["input_ids"], batch["attention_mask"])
            if args.type == 4:
                labels1 = labels1[0]
            if args.compressdim == -1:
                outputs = linear1(labels1)
                outputs1 = linear2(outputs)
            else:
                input1 = labels1.permute((0, 2, 1))
                outputs = linear1(input1)
                outputs1 = linear2(outputs)
                outputs1 = outputs1.permute((0, 2, 1))
            labels2 = part2(labels1, batch["attention_mask"])
            if args.compressdim == -1:
                outputs = linear3(labels2)
                outputs2 = linear4(outputs)
            else:
                input2 = labels2.permute((0, 2, 1))
                outputs = linear3(input2)
                outputs2 = linear4(outputs)
                outputs2 = outputs2.permute((0, 2, 1))
            loss1 = criterion(outputs1, labels1)
            loss2 = criterion(outputs2, labels2)
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if i % 20 == 0 and rank == 1:
                print("loss1", loss1, "loss2", loss2)
            if i % 300 == 0 and rank == 1:
                torch.save(
                    linear1.module.state_dict(),
                    str(args.task)
                    + "_"
                    + str(args.rank)
                    + "_"
                    + str(args.type)
                    + "_linear1.pth",
                )
                torch.save(
                    linear2.module.state_dict(),
                    str(args.task)
                    + "_"
                    + str(args.rank)
                    + "_"
                    + str(args.type)
                    + "_linear2.pth",
                )
                torch.save(
                    linear3.module.state_dict(),
                    str(args.task)
                    + "_"
                    + str(args.rank)
                    + "_"
                    + str(args.type)
                    + "_linear3.pth",
                )
                torch.save(
                    linear4.module.state_dict(),
                    str(args.task)
                    + "_"
                    + str(args.rank)
                    + "_"
                    + str(args.type)
                    + "_linear4.pth",
                )


if __name__ == "__main__":
    main()
