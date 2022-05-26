import torch.nn as nn
import time
import torch
from datasets import load_dataset
from transformers import ModelCard, get_scheduler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_metric
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import os
from models.models import *
from dataloader import create_dataloader_nlp
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
    PowerSVDLayerNLP,
)
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--dataset", default="rte", type=str)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--task", default="rte", type=str)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prun", default=0.0, type=float)
parser.add_argument("--kmeans", default=0, type=int)
parser.add_argument("--batches", default=32, type=int)
parser.add_argument("--sort", default=0, type=int)
parser.add_argument("--pca", default=0, type=int)
parser.add_argument("--type", default=1, type=int)
parser.add_argument("--linear", default=0, action="store_true")
parser.add_argument("--sortquant", default=0, action="store_true")
parser.add_argument("--fp16", default=0, action="store_true")
parser.add_argument("--eye", default=0, action="store_true")
parser.add_argument("--powersvd", default=0, type=int)
parser.add_argument("--powersvd1", default=0, type=int)
parser.add_argument("--poweriter", default=2, type=int)
parser.add_argument("--channelsize", default=100, type=int)
parser.add_argument("--worker", default=4, type=int)
parser.add_argument("--loader", default=12, type=int)
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
        backend="nccl",
        init_method="tcp://127.0.0.1:1237",
        world_size=args.worker,
        rank=rank,
    )
    # dataset dataloaer

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train_dataloader, val_dataloader, train_sampler = create_dataloader(args)
    # metric
    metric_mat = load_metric("glue", args.task)
    metric_acc = load_metric("accuracy")
    # model
    epochs = args.epochs
    if args.type == 1:
        model = RobertabaseLinear1(None, args.channelsize, args.eye)
    elif args.type == 2:
        model = RobertabaseLinear2(None, args.channelsize, args.eye)
    elif args.type == 3:
        model = RobertabaseLinear3(None, args.channelsize, args.eye)
    elif args.type == 4:
        model = RobertabaseLinear4(None, args.channelsize, args.eye)
    model = model.to(rank)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model = torch.nn.parallel.DistributedDataParallel(model)
    lr_scheduler = get_scheduler(
        name="polynomial",
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=epochs * len(train_dataloader),
    )
    print(len(train_dataloader))
    print(len(val_dataloader))
    criterion = nn.CrossEntropyLoss().to(rank)

    for epoch in range(epochs):
        model = model.train()
        train_loss = 0.0
        train_acc1 = 0.0
        time_avg = 0.0
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            start = time.time()
            batch = {k: v.to(rank) for k, v in batch.items()}
            # batch["attention_mask"] = torch.reshape(
            #     batch["attention_mask"],
            #     [
            #         int(batch["attention_mask"].shape[0]),
            #         1,
            #         1,
            #         int(batch["attention_mask"].shape[-1]),
            #     ],
            # ).to(rank)
            # batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e9

            # outputs = model(batch["input_ids"], batch["attention_mask"])
            outputs = model(batch["input_ids"], batch["attention_mask"])
            logits = outputs
            loss = criterion(logits, batch["labels"])
            pred = torch.argmax(logits, dim=1)
            acc = metric_acc.compute(predictions=pred, references=batch["labels"])
            loss.backward()
            optimizer.step()
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

        model.eval()
        metric_mat = load_metric("glue", args.task)
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                batch = {k: v.to(rank) for k, v in batch.items()}

                outputs = model(batch["input_ids"], batch["attention_mask"])

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
