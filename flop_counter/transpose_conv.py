import torch
import torch.nn as nn
from thop import profile

input = torch.rand([64, 32, 28, 28])
t_conv = nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
t_conv = nn.Sequential(*[t_conv])
mac1, param1 = profile(t_conv, inputs=(input,))
print(t_conv(input).shape)
print(mac1)
