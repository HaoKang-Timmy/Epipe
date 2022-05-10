from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
from utils import *
model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
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
train_dataset = load_dataset("glue", 'rte', split="train")
val_dataset = load_dataset("glue", 'rte', split="validation")
sentence1_key, sentence2_key = task_to_keys['rte']
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
train_dataset = train_dataset.map(
        lambda examples: {"labels": examples["label"]}, batched=True
    )
train_dataset.set_format(
        type="torch", columns=["input_ids", "labels", "attention_mask"]
    )

input = train_dataset[0]['input_ids'].view(1,-1)
mask = train_dataset[0]['attention_mask']
mask = (1.0 - mask) * -1e4

embedding = model.roberta.embeddings
attention = model.roberta.encoder.layer[0].attention
medium = model.roberta.encoder.layer[0].intermediate
output_layer = model.roberta.encoder.layer[0].output
roberta_layers = model.roberta.encoder.layer[1:]

part1 = EmbeddingAndAttention([embedding], [attention])
part2 = CombineLayer([medium], [output_layer], [roberta_layers])
part3 = model.roberta.encoder.layer[0]
part4 = nlp_sequential([model.roberta.encoder.layer[0:-1]])
input_ids = embedding(input)
output = part1(input,mask)
output1 = output[0].view(-1).detach().numpy()
output2 = part2(output,mask)
output2 = output2[0].view(-1).detach().numpy()
output3 = part3(input_ids,mask)
output3 = output3[0].view(-1).detach().numpy()
output4 = part4(input_ids,mask)
output4 = output4[0].view(-1).detach().numpy()
plt.subplot(2, 2, 1)
plt.hist(output1, bins=50, density=True)
plt.title("distribution(first attention layer)")
plt.xlabel("value")
plt.ylabel("number of values")
plt.subplot(2, 2, 2)
plt.hist(output2, bins=50, density=True)
plt.title("distribution(last layer)")
plt.xlabel("value")
plt.ylabel("number of values")
plt.subplot(2, 2, 3)
plt.hist(output3, bins=50, density=True)
plt.title("distribution(first layer)")
plt.xlabel("value")
plt.ylabel("number of values")
plt.subplot(2, 2, 4)
plt.hist(output4, bins=50, density=True)
plt.title("distribution(second last layer)")
plt.xlabel("value")
plt.ylabel("number of values")
plt.tight_layout()
plt.show()

