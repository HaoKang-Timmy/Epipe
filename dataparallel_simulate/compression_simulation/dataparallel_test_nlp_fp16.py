import torch.nn as nn
import time
import torch
from datasets import load_dataset
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_metric
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import os
from utils import (
    Fakequantize,
    TopkLayer,
    SortQuantization,
    KMeansLayer,
    PCAQuantize,
    combine_classifier,
    combine_embeding,
    CombineLayer,
    EmbeddingAndAttention,
)
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast


class nlp_sequential(nn.Module):
    def __init__(self, layers: list):
        super(nlp_sequential, self).__init__()
        self.layers = layers[0]

    def forward(self, output: torch.tensor, mask: torch.tensor):
        for i, layer in enumerate(self.layers):
            output = layer(output, mask)
            output = output[0]
        return output


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--log", default="./my_gpipe", type=str)
parser.add_argument("--dataset", default="rte", type=str)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--task", default="rte", type=str)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prun", default=0.0, type=float)
parser.add_argument("--kmeans", default=0, type=int)
parser.add_argument("--batches", default=8, type=int)
parser.add_argument("--sort", default=0, type=int)
parser.add_argument("--pca", default=0, type=int)
parser.add_argument("--linear", default=0, action="store_true")
parser.add_argument("--sortquant", default=0, action="store_true")
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=4, args=(4, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:1237", world_size=4, rank=rank
    )
    # dataset dataloaer

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train_dataset = load_dataset("glue", args.task, split="train")
    val_dataset = load_dataset("glue", args.task, split="validation")
    sentence1_key, sentence2_key = task_to_keys[args.task]
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    # sentence1_key, sentence2_key = task_to_keys["cola"]
    def encode(examples):
        if sentence2_key is not None:
            return tokenizer(
                examples[sentence1_key],
                examples[sentence2_key],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        return tokenizer(
            examples[sentence1_key],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_dataset = train_dataset.map(encode, batched=True)
    val_dataset = val_dataset.map(encode, batched=True)
    val_dataset = val_dataset.map(
        lambda examples: {"labels": examples["label"]}, batched=True
    )
    train_dataset = train_dataset.map(
        lambda examples: {"labels": examples["label"]}, batched=True
    )
    train_dataset.set_format(
        type="torch", columns=["input_ids", "labels", "attention_mask"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "labels", "attention_mask"]
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        sampler=train_sampler,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )

    # metric
    metric_mat = load_metric("glue", args.task)
    metric_acc = load_metric("accuracy")
    # model
    epochs = args.epochs
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
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

    if args.linear != 0:
        linear1 = torch.nn.Linear(768, 224).to(rank)
        linear2 = torch.nn.Linear(224, 768).to(rank)
        linear3 = torch.nn.Linear(768, 224).to(rank)
        linear4 = torch.nn.Linear(224, 768).to(rank)

    if args.linear == 0:
        optimizer = AdamW(
            [
                {"params": part1.parameters()},
                {"params": part2.parameters()},
                {"params": part3.parameters()},
            ],
            lr=args.lr,
        )
    else:
        optimizer = AdamW(
            [
                {"params": part1.parameters()},
                {"params": part2.parameters()},
                {"params": part3.parameters()},
                {"params": linear1.parameters(), "lr": args.lr},
                {"params": linear2.parameters(), "lr": args.lr},
                {"params": linear3.parameters(), "lr": args.lr},
                {"params": linear4.parameters(), "lr": args.lr},
            ],
            lr=args.lr,
        )

    part1 = torch.nn.parallel.DistributedDataParallel(part1)
    part2 = torch.nn.parallel.DistributedDataParallel(part2)
    part3 = torch.nn.parallel.DistributedDataParallel(part3)
    lr_scheduler = get_scheduler(
        name="polynomial",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=epochs * len(train_dataloader),
    )
    print(len(train_dataloader))
    print(len(val_dataloader))
    criterion = nn.CrossEntropyLoss().to(rank)
    topk_layer = TopkLayer(args.prun).to(rank)
    scaler = GradScaler()
    for epoch in range(epochs):
        part1.train()
        part2.train()
        part3.train()
        train_loss = 0.0
        train_acc1 = 0.0
        time_avg = 0.0
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            start = time.time()
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
                .type(torch.float16)
            )
            batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e4

            with autocast():
                outputs = part1(batch["input_ids"], batch["attention_mask"])

                # loss = outputs.loss
                # print(outputs.shape)

                if args.prun != 0:
                    outputs = topk_layer(outputs)
                if args.quant != 0:
                    outputs = Fakequantize.apply(outputs, args.quant)
                # if args.kmeans != 0:
                #     outputs = kmeanslayer(outputs)
                # if rank == 1:
                # print("first output",outputs)
                if args.pca != 0:
                    outputs = PCAQuantize.apply(outputs, args.pca)
                if args.linear != 0:
                    outputs = linear1(outputs)
                    outputs = linear2(outputs)
                if args.sortquant != 0:
                    outputs = SortQuantization.apply(
                        outputs, args.quant, args.prun, args.sort
                    )
                outputs = part2(outputs, batch["attention_mask"])
                print(outputs.dtype)
                if args.prun != 0:
                    outputs = topk_layer(outputs)
                if args.quant != 0:
                    outputs = Fakequantize.apply(outputs, args.quant)
                # if args.kmeans != 0:
                #     outputs = kmeanslayer(outputs)
                # if rank == 1:
                #     print("first output",outputs)
                if args.pca != 0:
                    outputs = PCAQuantize.apply(outputs, args.pca)
                if args.linear != 0:
                    outputs = linear3(outputs)
                    outputs = linear4(outputs)
                if args.sortquant != 0:
                    outputs = SortQuantization.apply(
                        outputs, args.quant, args.prun, args.sort
                    )

                outputs = part3(outputs)
                logits = outputs
                # batch["labels"] = batch["labels"].type(torch.float16)
                loss = criterion(logits, batch["labels"])
                pred = torch.argmax(logits, dim=1)
                acc = metric_acc.compute(predictions=pred, references=batch["labels"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_loss += loss.item()
            train_acc1 += acc["accuracy"]

            end = time.time() - start
            time_avg += end
            if i % 20 == 0 and rank == 1:
                print("train_loss", loss.item(), "train_acc", acc["accuracy"])
        train_loss /= len(train_dataloader)
        train_acc1 /= len(train_dataloader)
        time_avg /= len(train_dataloader)
        lr_scheduler.step()
        val_loss = 0.0
        val_matt = 0.0
        val_acc1 = 0.0

        part1.eval()
        part2.eval()
        part3.eval()
        metric_mat = load_metric("glue", args.task)
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
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
                    .type(torch.float16)
                )
                batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e9
                outputs = part1(batch["input_ids"], batch["attention_mask"])

                # print(outputs)
                # loss = outputs.loss

                if args.prun != 0:
                    outputs = topk_layer(outputs)
                if args.quant != 0:
                    outputs = Fakequantize.apply(outputs, args.quant)
                # if args.kmeans != 0:
                #     outputs = kmeanslayer(outputs)
                if args.pca != 0:
                    outputs = PCAQuantize.apply(outputs, args.pca)
                if args.linear != 0:
                    outputs = linear1(outputs)
                    outputs = linear2(outputs)
                if args.sortquant != 0:
                    outputs = SortQuantization.apply(
                        outputs, args.quant, args.prun, args.sort
                    )
                outputs = part2(outputs, batch["attention_mask"])

                if args.prun != 0:
                    outputs = topk_layer(outputs)
                if args.quant != 0:
                    outputs = Fakequantize.apply(outputs, args.quant)
                # if args.kmeans != 0:
                #     outputs = kmeanslayer(outputs)
                if args.pca != 0:
                    outputs = PCAQuantize.apply(outputs, args.pca)
                if args.linear != 0:
                    outputs = linear3(outputs)
                    outputs = linear4(outputs)
                if args.sortquant != 0:
                    outputs = SortQuantization.apply(
                        outputs, args.quant, args.prun, args.sort
                    )
                outputs = part3(outputs)
                logits = outputs
                loss = criterion(logits, batch["labels"])

                pred = torch.argmax(logits, dim=1)
                acc = metric_acc.compute(predictions=pred, references=batch["labels"])
                metric_mat.add_batch(predictions=pred, references=batch["labels"])
                val_loss += loss.item()
                val_acc1 += acc["accuracy"]
                if i % 20 == 0 and rank == 1:
                    print("val_loss", loss.item(), "val_acc", acc["accuracy"], "matt")
            val_loss /= len(val_dataloader)
            val_acc1 /= len(val_dataloader)
            val_matt = metric_mat.compute()

        if rank == 1:
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
                "matt",
                val_matt,
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
                + "  matthew:"
                + str(val_matt)
            )
            file_save.close()


if __name__ == "__main__":
    main()
