from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

train_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)


def encode(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


train_dataset = train_dataset.map(encode, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
