'''
Author: your name
Date: 2022-04-03 02:05:05
LastEditTime: 2022-04-03 10:38:14
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe/compression/sparse_layer.py
'''
from os import stat
import torch.nn.functional as F
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist

def create_sparse(input:torch.tensor,bit_saving = True):
    shape = input.shape
    input = input.view(-1)
    index = input.nonzero()
    index = index.view(-1)
    if bit_saving is True:
        index = index.type(torch.bfloat16)
    src = input.index_select(0,index)
    return shape,index,src
def unzip_sparse(input,index,src,shape):
    input = input.view(-1)
    input.scatter_(0,index,src)
    input = input.view(shape)
    return input


