from datasets import load_dataset
dataset = load_dataset("glue","cola",split='train')
from transformers import AutoModelForSequenceClassification, AutoTokenizer,
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def encode(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length')

dataset = dataset.map(encode, batched=True)
dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
import torch
dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
# print(next(iter(dataloader)))