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

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--log", default="./my_gpipe", type=str)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--task", default="rte", type=str)
parser.add_argument("--batches", default=8, type=int)
parser.add_argument("--rank", default=100, type=int)
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


class EmbeddingAndAttention(nn.Module):
    def __init__(self, embedding_layer, attention_layer):
        super(EmbeddingAndAttention, self).__init__()
        self.embedding_layer = embedding_layer[0]
        self.attention_layer = attention_layer[0]

    def forward(self, input, mask):
        output = self.embedding_layer(input)
        output = self.attention_layer(output, mask)
        return output[0]


class CombineLayer(nn.Module):
    def __init__(self, mediumlayer, output_layer, others):
        super(CombineLayer, self).__init__()
        self.medium = mediumlayer[0]
        self.outputlayer = output_layer[0]
        self.others = others[0]

    def forward(self, input, mask):
        output = self.medium(input)
        output = self.outputlayer(output, input)
        for i, layer in enumerate(self.others):
            output = layer(output, mask)
            output = output[0]
        return output


def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=4, args=(4, args))


def main_worker(rank, process_num, args):
    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:1237", world_size=4, rank=rank
    )
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
    metric_mat = load_metric("glue", args.task)
    metric_acc = load_metric("accuracy")
    epochs = args.epochs
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    for param in model.parameters():
        param.require_grad = False
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

    linear1 = nn.Linear(768, args.rank).to(rank)
    linear2 = nn.Linear(args.rank, 768).to(rank)
    linear3 = nn.Linear(768, args.rank).to(rank)
    linear4 = nn.Linear(args.rank, 768).to(rank)
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
        num_warmup_steps=100,
        num_training_steps=epochs * len(train_dataloader),
    )
    criterion = nn.MSELoss().to(rank)
    for epoch in range(epochs):
        part1.eval()
        part2.eval()
        part3.eval()
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
            labels1 = part1(batch["input_ids"], batch["attention_mask"])
            outputs = linear1(labels1)
            outputs1 = linear2(outputs)
            labels2 = part2(labels1, batch["attention_mask"])
            outputs = linear3(labels2)
            outputs2 = linear4(outputs)
            loss1 = criterion(outputs1, labels1)
            loss2 = criterion(outputs2, labels2)
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if i % 20 == 0 and rank == 1:
                print("loss1", loss1, "loss2", loss2)
            torch.save(linear1.module.state_dict(), str(args.rank) + "_linear1.pty")
            torch.save(linear2.module.state_dict(), str(args.rank) + "_linear2.pty")
            torch.save(linear3.module.state_dict(), str(args.rank) + "_linear3.pty")
            torch.save(linear4.module.state_dict(), str(args.rank) + "_linear4.pty")


if __name__ == "__main__":
    main()
