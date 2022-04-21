from asyncio import transports

from numpy import transpose
from utils import FakeQuantize, SortQuantization, KMeansLayer, PCAQuantize
from ignite.contrib.metrics.regression.mean_absolute_relative_error import (
    MeanAbsoluteRelativeError,
)
import torch
import torch.nn as nn
import argparse
import math
import time

parser = argparse.ArgumentParser(description="Compression method test")
parser.add_argument("--tensorsize", default=1000000, type=int)
parser.add_argument("--matrixsize", default=[5, 1000, 1000], type=list)
parser.add_argument("--split", default=2, type=int)
parser.add_argument("--sortquant", default=6, type=int)
parser.add_argument("--quant", default=8, type=int)
parser.add_argument("--kmeans", default=8, type=int)
parser.add_argument("--rank", default=125, type=int)
args = parser.parse_args()
metric = MeanAbsoluteRelativeError()
kmeans_layer = KMeansLayer(args.kmeans, 0)
input = torch.rand([args.tensorsize]).to(0)
s_time = time.time()
s_output = SortQuantization.apply(input, 6, 2)
s_time = time.time() - s_time
k_time = time.time()
k_output = kmeans_layer(input)
k_time = time.time() - k_time
q_time = time.time()
q_output = FakeQuantize.apply(input, 8)
q_time = time.time() - q_time
matrix = torch.rand(args.matrixsize)
p_time = time.time()
p_output = PCAQuantize.apply(matrix, args.rank)
p_time = time.time() - p_time
metric.reset
metric._update([input, q_output])
print("uniform quant MARE loss:", metric.compute(), "uniform quant MARE time:", q_time)
metric.reset
metric._update([input, s_output])
print("sort quant MARE loss:", metric.compute(), "sort quant MARE time:", s_time)
metric.reset
metric._update([input, k_output])
print("kmeans quant MARE loss:", metric.compute(), "kmeans MARE time:", k_time)
matrix = matrix.view(-1)
p_output = p_output.view(-1)
metric.reset
metric._update([matrix, p_output])
print("PCA MARE loss:", metric.compute(), "PCA MARE time:", p_time)
