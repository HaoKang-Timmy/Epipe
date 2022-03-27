'''
Author: your name
Date: 2022-03-26 02:28:43
LastEditTime: 2022-03-26 15:37:29
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/test/compression_test.py
'''
import torch
from utils import QuantizationLayer,DequantizationLayer,Fakequantize
import torch.nn as nn
avgpool = nn.AvgPool2d((2,2))
up = nn.Upsample(scale_factor=2, mode='bilinear')
soth = torch.rand([64,32,32,32])
# quant_layer = QuantizationLayer(8)
soth1 =avgpool(soth)
# dequant_layer = DequantizationLayer(8)
# soth1,min,step = quant_layer(soth)
# print(soth.max())
# print("min",min,"step",step)
# print(soth)
# print(soth1)
# soth2 = dequant_layer(soth1,min,step,quant_layer.backward_min,quant_layer.backward_step)
soth2 = Fakequantize.apply(soth1,32)
soth2 = up(soth2)
print(torch.abs(torch.abs(soth)-torch.abs(soth2)).sum()/torch.abs(soth).sum())
