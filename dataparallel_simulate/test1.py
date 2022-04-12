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
import torch

input = torch.rand([4, 4])
input = input.chunk(2)
print(input)
input = torch.cat(input, 0)
print(input)
# print(input[0])
