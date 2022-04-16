"""
Author: your name
Date: 2022-03-18 16:58:49
LastEditTime: 2022-04-12 22:00:40
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dataparallel_simulate/test1.py
"""
"""
Author: your name
Date: 2022-03-18 16:58:49
LastEditTime: 2022-03-29 13:07:31
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/test/test1.py
"""
# from datasets import load_dataset
# dataset = load_dataset("glue","cola",split='train')
# from transformers import AutoModelForSequenceClassification, AutoTokenizer,
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# def encode(examples):
#     return tokenizer(examples['sentence'], truncation=True, padding='max_length')

# dataset = dataset.map(encode, batched=True)
# dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
# import torch
# dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
# print(next(iter(dataloader)))
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# from dist_gpipe_gloo import dist_gpipe,Reshape1
from torchvision.models import mobilenet_v2
import torch.nn as nn
import argparse
import torch
import os


def main():
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
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train_dataset = load_dataset("glue", "rte", split="train")
    val_dataset = load_dataset("glue", "rte", split="validation")
    sentence1_key, sentence2_key = task_to_keys["rte"]
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
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        # sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )
    for i, batch in enumerate(train_loader):
        print(batch["input_ids"])
        print(batch["attention_mask"])
        break


if __name__ == "__main__":
    main()
