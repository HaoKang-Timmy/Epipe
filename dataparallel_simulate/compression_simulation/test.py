"""
Author: your name
Date: 2022-04-21 22:04:09
LastEditTime: 2022-05-02 01:11:13
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE

FilePath: /research/gpipe_test/dataparallel_simulate/compression_simulation/test.py
"""
import torch
from utils import SortQuantization

input = torch.rand([10, 10])
some = input > 0
print(some)
