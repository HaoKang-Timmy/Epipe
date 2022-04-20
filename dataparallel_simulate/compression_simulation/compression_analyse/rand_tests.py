from sklearn.metrics import mean_poisson_deviance
from utils import FakeQuantize, SortQuantization, KMeansLayer
from ignite.contrib.metrics.regression.mean_absolute_relative_error import (
    MeanAbsoluteRelativeError,
)
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description="Compression method test")
parser.add_argument("--tensorsize", default=1000000, type=int)
parser.add_argument("--split", default=2, type=int)
parser.add_argument("--sortquant", default=6, type=int)
parser.add_argument("--quant", default=8, type=int)
parser.add_argument("--kmeans", default=8, type=int)
args = parser.parse_args()
metric = MeanAbsoluteRelativeError()
kmeans_layer = KMeansLayer(args.kmeans, 0)
input = torch.rand([args.tensorsize]).to(0)
s_output = SortQuantization.apply(input, 6, 2)
k_output = kmeans_layer(input)
q_output = FakeQuantize.apply(input, 8)
metric.reset
metric._update([input, q_output])
print("uniform quant MARE loss:", metric.compute())
metric.reset
metric._update([input, s_output])
print("sort quant MARE loss:", metric.compute())
metric.reset
metric._update([input, k_output])
print("kmeans quant MARE loss:", metric.compute())
