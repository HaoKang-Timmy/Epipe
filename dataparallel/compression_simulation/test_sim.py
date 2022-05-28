from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.models import *
from datasets import load_dataset
import os

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

model1 = AutoModelForSequenceClassification.from_pretrained("roberta-base")
model2 = Robertabase(model1)
model1.eval()
model2.eval()

train_dataset = load_dataset("glue", "rte", split="train")
val_dataset = load_dataset("glue", "rte", split="validation")
sentence1_key, sentence2_key = task_to_keys["rte"]
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
val_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    num_workers=12,
    pin_memory=True,
    drop_last=True,
    shuffle=False,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=8,
    num_workers=12,
    pin_memory=True,
    drop_last=True,
    shuffle=False,
)
model1 = model1.to(1)
model2 = model2.to(1)
for i, batch in enumerate(train_dataloader):
    batch = {k: v.to(1) for k, v in batch.items()}
    output1 = model1(batch["input_ids"], batch["attention_mask"])
    output2 = model2(batch["input_ids"], batch["attention_mask"])
    break
print(output1)
print(output2)
