import torch
import torch.nn as nn
from thop import profile

input = torch.rand([64, 32, 28, 28])
t_conv = nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
t_conv = nn.Sequential(*[t_conv])
mac1, param1 = profile(t_conv, inputs=(input,))
print(t_conv(input).shape)
print(mac1)
input = torch.rand([64, 32, 112, 112])
conv = nn.Conv2d(32, 32, (4, 4), (4, 4))
conv = nn.Sequential(*[conv])
mac1, param1 = profile(conv, inputs=(input,))
print(conv(input).shape)
print(mac1)
