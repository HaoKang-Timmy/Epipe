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
from torch.optim import AdamW

# layer for partition and training
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
parser.add_argument("--log-dir", default="./my_gpipe", type=str)
parser.add_argument("--lr", default=4e-5, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--worker", default=4, type=int)
parser.add_argument("--task", default="rte", type=str)

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
    mp.spawn(main_worker, nprocs=args.worker, args=(args.worker, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:1247",
        world_size=args.worker,
        rank=rank,
    )
    # dataset dataloaer
    print("rank", rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train_dataset = load_dataset("glue", args.task, split="train")
    val_dataset = load_dataset("glue", args.task, split="validation")
    sentence1_key, sentence2_key = task_to_keys[args.task]
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

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
        batch_size=16,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        sampler=train_sampler,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
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
    model1 = model.roberta.embeddings
    model2 = nlp_sequential([model.roberta.encoder.layer[0:]])
    model3 = model.classifier
    model1 = model1.to(rank)
    model2 = model2.to(rank)
    model3 = model3.to(rank)
    model1 = torch.nn.parallel.DistributedDataParallel(model1)
    model2 = torch.nn.parallel.DistributedDataParallel(model2)
    model3 = torch.nn.parallel.DistributedDataParallel(model3)
    optimizer = AdamW(
        [
            {"params": model1.parameters()},
            {"params": model2.parameters()},
            {"params": model3.parameters()},
        ],
        lr=args.lr,
    )
    lr_scheduler = get_scheduler(
        name="polynomial",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=epochs * len(train_dataloader),
    )
    print(len(train_dataloader))
    print(len(val_dataloader))
    criterion = nn.CrossEntropyLoss().to(rank)
    # topk_layer = TopkLayer(args.prun).to(rank)
    for epoch in range(epochs):

        model1.train()
        model2.train()
        model3.train()
        train_loss = 0.0
        train_acc1 = 0.0
        time_avg = 0.0
        train_sampler.set_epoch(epoch)
        start = time.time()
        for i, batch in enumerate(train_dataloader):

            batch = {k: v.to(rank) for k, v in batch.items()}
            optimizer.zero_grad()
            # outputs = model(batch['input_ids'],)
            output = model1(batch["input_ids"])
            batch["attention_mask"] = torch.reshape(
                batch["attention_mask"],
                [
                    int(batch["attention_mask"].shape[0]),
                    1,
                    1,
                    int(batch["attention_mask"].shape[-1]),
                ],
            ).to(rank)
            batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e4
            output = model2(output, batch["attention_mask"])
            output = model3(output)
            logits = output
            # print(logits)
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
            start = time.time()
        train_loss /= len(train_dataloader)
        train_acc1 /= len(train_dataloader)
        time_avg /= len(train_dataloader)
        lr_scheduler.step()
        val_loss = 0.0
        val_matt = 0.0
        val_acc1 = 0.0

        model1.eval()
        model2.eval()
        model3.eval()
        metric_mat = load_metric("glue", args.task)
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                batch = {k: v.to(rank) for k, v in batch.items()}
                output = model1(batch["input_ids"])
                batch["attention_mask"] = torch.reshape(
                    batch["attention_mask"],
                    [
                        int(batch["attention_mask"].shape[0]),
                        1,
                        1,
                        int(batch["attention_mask"].shape[-1]),
                    ],
                ).to(rank)
                batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e4
                output = model2(output, batch["attention_mask"])
                output = model3(output)
                logits = output
                loss = criterion(logits, batch["labels"])
                # logits = outputs.logits
                # acc,_ = accuracy(logits,batch['labels'],topk =(1,2))
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
                "matt",
                val_matt,
            )
            file_save = open(args.log_dir, mode="a")
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
