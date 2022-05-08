"""
Author: your name
Date: 2022-04-12 16:29:18
LastEditTime: 2022-04-12 21:36:19
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/test_vision_dgpipe.py
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from dist_gpipe_gloo import (
    dist_gpipe,
    Reshape1,
    nlp_sequential,
    combine_classifier,
    combine_embeding,
)
from torchvision.models import mobilenet_v2
import torch.nn as nn
import argparse
import torch
import os

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--chunks", default=4, type=int)
parser.add_argument("--log", default="./nlp_test.txt", type=str)
parser.add_argument("--train-method", default="finetune", type=str)
# parser.add_argument("--warmup", default=0, action="store_true")
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--wd", default=0.0, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--batches", default=32, type=int)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prune", default=0.0, type=float)
parser.add_argument("--world-size", default=2, type=int)
parser.add_argument("--showperiod", default=20, type=int)
parser.add_argument("--tasktype", default="nlp", type=str)
parser.add_argument("--root", default="../data", type=str)
parser.add_argument("--devices", default=[0, 1], type=list)
parser.add_argument("--url", default="tcp://127.0.0.1:1224", type=str)
parser.add_argument("--bachend", default="nccl", type=str)
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--sortquant", default=0, action="store_true")
parser.add_argument("--task", default="rte", type=str)
parser.add_argument("--fastquant", default=0, action="store_true")
parser.add_argument("--bandwidth", default=0, action="store_true")
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
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batches,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        # sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batches,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    model1 = [model.roberta.embeddings]
    model7 = nlp_sequential([model.roberta.encoder.layer[0:1]])
    model2 = nlp_sequential([model.roberta.encoder.layer[1:-1]])

    model5 = nlp_sequential([model.roberta.encoder.layer[-1:]])
    model6 = model.classifier
    model5 = combine_classifier([model5], [model6])
    # model1 = nn.Sequential(*model1)
    model1 = combine_embeding([model7], model1)
    # model4 = nn.Sequential(*model4)
    devices = args.devices

    partition = [[model1, model5], [model2]]
    tensor_size = [
        [
            (int(args.batches / args.chunks), 128, 768),
            (int(args.batches / args.chunks), 128, 768),
        ],
        [
            (int(args.batches / args.chunks), 128, 768),
            (int(args.batches / args.chunks), 128, 768),
        ],
    ]
    print(tensor_size)
    model = dist_gpipe(args, partition, devices, tensor_size, train_loader, val_loader)
    model.session()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
