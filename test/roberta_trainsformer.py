"""
Author: your name
Date: 2022-03-18 00:24:21
LastEditTime: 2022-03-19 00:20:02
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/test/roberta_trainsformer.py
"""
import torch
from transformers import (
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    DataCollatorWithPadding,
)
import torchvision.transforms as transforms
from datasets import load_dataset

# model
def main():
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    # model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
    # tokenizer
    print(model)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # dataset
    datasets = load_dataset("glue", "cola", split="train")

    def preprocess_func(examples):
        token = tokenizer(examples["sentence"], truncation=True, padding="max_length",)
        # print(token['input_ids'])
        # token['input_ids'] = torch.tensor(token['input_ids'])
        # token['attention_mask'] = torch.tensor(token['attention_mask'])
        return token

    tokenized_cola = datasets.map(preprocess_func, batched=True)
    tokenized_cola = tokenized_cola.map(
        lambda examples: {"labels": examples["label"]}, batched=True
    )

    tokenized_cola.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    # print(tokenized_cola[0])
    dataloader = torch.utils.data.DataLoader(
        tokenized_cola, batch_size=128, shuffle=True
    )
    # print(next(iter(dataloader)))
    # for batch_data in dataloader:
    #     print(batch_data)


# dataloader
# data_collator = DataCollatorWithPadding(tokenizer = tokenizer,padding=True)

# for dict in
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# print(tokenizer)
# output = tokenizer(train_dataset[2]['sentence'],return_tensors = "pt")
# # print(train_dataset[2]['sentence'])
# # print(output['input_ids'])
# print(output)
# # dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = 512)
# output = model(output['input_ids'])
# print(output)
# for things in dataloader:
#     print(things)
# output = model(train_dataset[2]['sentence'])
# print(output)
if __name__ == "__main__":
    main()
