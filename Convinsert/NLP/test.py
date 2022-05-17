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

model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
for param in model.parameters():
    param.require_grad = False
