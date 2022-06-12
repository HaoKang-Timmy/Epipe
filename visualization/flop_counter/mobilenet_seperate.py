from models import MobileNetV2withConvInsert3
from torchvision.models import mobilenet_v2
from thop import profile
import torch.nn as nn
import torch

input1 = torch.rand([64, 3, 224, 224])
input2 = torch.rand([64, 16, 28, 28])
input3 = torch.rand([64, 1280, 7, 7])
input4 = torch.rand([64, 20, 37, 37])
input5 = torch.rand([64, 100, 7, 7])
model = MobileNetV2withConvInsert3()
# reshape = model.reshape

part11 = model.mobilenetv2_part1
part12 = model.conv1
part13 = model.conv2
part1 = nn.Sequential(*[part11, part12, part13])
part21 = model.t_conv2
part22 = model.t_conv1
part23 = model.mobilenetv2_part2
part24 = model.conv3
part25 = model.conv4
part2 = nn.Sequential(*[part21, part22, part23, part24, part25])
part31 = model.t_conv4
part32 = model.t_conv3
part33 = model.reshape
part34 = model.mobilenetv2_part3
part3 = nn.Sequential(*[part31, part32, part33, part34])
mac1, param1 = profile(part1, inputs=(input1,))
print(mac1)
mac2, param2 = profile(part2, inputs=(input2,))
print(mac2)
mac3, param3 = profile(part3, inputs=(input5,))
print(mac3)
