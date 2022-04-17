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


class combine_embeding(nn.Module):
    def __init__(self, layers: list, embed_layer):
        super(combine_embeding, self).__init__()
        self.layers = layers[0]
        self.embed_layer = embed_layer[0]

    def forward(self, input: torch.tensor, mask: torch.tensor):
        output = self.embed_layer(input)

        output = self.layers(output, mask)
        output = output
        return output


class combine_classifier(nn.Module):
    def __init__(self, layers: list, classifier):
        super(combine_classifier, self).__init__()
        self.layers = layers[0]
        self.classifier = classifier[0]

    def forward(self, output: torch.tensor, mask: torch.tensor):
        # for i, layer in enumerate(self.layers):
        output = self.layers(output, mask)
        output = output
        output = self.classifier(output)
        return output


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--root", default="./data", type=str)
parser.add_argument("--log", default="./test.txt", type=str)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--worker", default=4, type=int)
parser.add_argument("--task", default="rte", type=str)
parser.add_argument("--batches", default=16, type=int)

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
        batch_size=args.batches,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        sampler=train_sampler,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batches,
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
    model1 = [model.roberta.embeddings]
    model2 = nlp_sequential([model.roberta.encoder.layer[0:1]])
    model3 = nlp_sequential([model.roberta.encoder.layer[1:-1]])
    model4 = nlp_sequential([model.roberta.encoder.layer[-1:]])
    model5 = model.classifier
    part1 = combine_embeding([model2], model1)
    part2 = model3
    part3 = combine_classifier([model4], [model5])

    part1 = part1.to(rank)
    part2 = part2.to(rank)
    part3 = part3.to(rank)
    part1 = torch.nn.parallel.DistributedDataParallel(part1)
    part2 = torch.nn.parallel.DistributedDataParallel(part2)
    part3 = torch.nn.parallel.DistributedDataParallel(part3)
    optimizer = AdamW(
        [
            {"params": part1.parameters()},
            {"params": part2.parameters()},
            {"params": part3.parameters()},
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

        part1.train()
        part2.train()
        part3.train()
        train_loss = 0.0
        train_acc1 = 0.0
        time_avg = 0.0
        datatime_avg = 0.0
        backward_avg = 0.0
        train_sampler.set_epoch(epoch)
        start = time.time()
        for i, batch in enumerate(train_dataloader):

            batch = {k: v.to(rank, non_blocking=True) for k, v in batch.items()}
            datatime = time.time() - start
            # print("datatime:",datatime)
            datatime_avg += datatime

            # outputs = model(batch['input_ids'],)
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
            batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e4
            output = part1(batch["input_ids"], batch["attention_mask"])
            # print(output.shape)

            output = part2(output, batch["attention_mask"])
            # print(output.shape)
            output = part3(output, batch["attention_mask"])
            # print(output.shape)
            logits = output
            # print(logits)
            loss = criterion(logits, batch["labels"])
            pred = torch.argmax(logits, dim=1)
            acc = metric_acc.compute(predictions=pred, references=batch["labels"])
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            train_loss += loss.item()
            train_acc1 += acc["accuracy"]

            end = time.time() - start
            backwardend = time.time() - backward_start
            backward_avg += backwardend
            time_avg += end
            if i % 20 == 0 and rank == 1:
                print("train_loss", loss.item(), "train_acc", acc["accuracy"])
            start = time.time()
        train_loss /= len(train_dataloader)
        train_acc1 /= len(train_dataloader)
        time_avg /= len(train_dataloader)
        datatime_avg /= len(train_dataloader)
        backward_avg /= len(train_dataloader)
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
                    .type(torch.float32)
                )
                batch["attention_mask"] = (1.0 - batch["attention_mask"]) * -1e4
                output = part1(batch["input_ids"], batch["attention_mask"])

                output = part2(output, batch["attention_mask"])
                output = part3(output, batch["attention_mask"])
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
                + "  datatime:"
                + str(datatime_avg)
                + "  backward_time:"
                + str(backward_avg)
                + "  matthew:"
                + str(val_matt)
            )
            file_save.close()


if __name__ == "__main__":
    main()
