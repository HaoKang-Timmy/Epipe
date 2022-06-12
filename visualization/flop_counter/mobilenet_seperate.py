
from models import MobileNetV2withConvInsert3, MobileNetV2withConvInsert2
from torchvision.models import mobilenet_v2
from thop import profile
from thop import clever_format
import torch.nn as nn
import torch

input1 = torch.rand([1, 3, 224, 224])
input2 = torch.rand([1, 16, 28, 28])
input3 = torch.rand([1, 100, 7, 7])

model = MobileNetV2withConvInsert3()
# reshape = model.reshape

part11 = model.mobilenetv2_part1
part12 = model.conv1
part13 = model.conv2
part1 = nn.Sequential(*[part11,part12,part13])
part21 = model.t_conv2
part22 = model.t_conv1
part23 = model.mobilenetv2_part2
part24 = model.conv3
part25 = model.conv4
part2 = nn.Sequential(*[part21,part22,part23,part24,part25])
part31 = model.t_conv4
part32 = model.t_conv3
part33 = model.reshape 
part34 = model.mobilenetv2_part3
part3 = nn.Sequential(*[part31, part32,part33,part34])
mac1, param1 = profile(part1, inputs=(input1,))
mac1, param1 = clever_format([mac1, param1], "%.3f")
print(mac1,param1)
mac2, param2 = profile(part2, inputs=(input2,))
mac2, param2 = clever_format([mac2, param2], "%.3f")
print(mac2,param2)
mac3, param3 = profile(part3, inputs=(input3,))
mac3, param3 = clever_format([mac3, param3], "%.3f")
print(mac3,param3)
