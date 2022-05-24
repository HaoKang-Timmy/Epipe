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

linear = torch.nn.Linear(768, 100)
torch.save(linear.state_dict(), "./save_linear")
linear.load_state_dict(torch.load("./save_linear"))
