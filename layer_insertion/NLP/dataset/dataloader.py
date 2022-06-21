from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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


def create_dataloader(args):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    if args.task != "wiki":
        train_dataset = load_dataset("glue", args.task, split="train")
        val_dataset = load_dataset("glue", args.task, split="validation")
        sentence1_key, sentence2_key = task_to_keys[args.task]
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
        train_dataset = load_dataset("glue", args.task, split="train")
        val_dataset = load_dataset("glue", args.task, split="validation")
        sentence1_key, sentence2_key = task_to_keys[args.task]

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
    else:
        train_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        val_dataset = load_dataset("wikitext", "wikitext-2-v1", split="validation")

        def encode(examples):
            return tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=128,
            )

        train_dataset = train_dataset.map(encode, batched=True)
        val_dataset = val_dataset.map(encode, batched=True)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
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
    return train_dataloader, val_dataloader, train_sampler
