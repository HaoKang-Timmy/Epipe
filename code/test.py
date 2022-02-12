from models import MobileNetV2
from distributed_layers import Reshape
import torch.nn as nn
import torch
model = MobileNetV2()
# print(model)
layer1 = nn.Sequential(model.conv1,model.bn1,)
layer2 = nn.Sequential(model.layers[0:6])
layer3 = nn.Sequential(model.layers[6:13])
layer4 = nn.Sequential(model.layers[13:],model.conv2,model.bn2,nn.AvgPool2d(4),Reshape())
layer5 = nn.Sequential(model.linear)
input = torch.rand([128,3,32,32])
output = layer1(input)
output = layer2(output)
output = layer3(output)
output = layer4(output)
partition = [layer1,layer2,layer3,layer4,layer5]
print(output.shape)

