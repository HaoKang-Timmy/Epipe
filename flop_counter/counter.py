'''
Author: Beta Cat 466904389@qq.com
Date: 2022-06-23 22:02:03
LastEditors: Beta Cat 466904389@qq.com
LastEditTime: 2022-06-25 15:28:57
FilePath: /research/gpipe_test/flop_counter/counter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from models import *
from thop import profile,clever_format
import argparse
parser = argparse.ArgumentParser(description="Flop counter of models")
parser.add_argument("--type", default=0, type=int)
args = parser.parse_args()
if args.type == 0:
    model = MobileNetV2withConvInsert0_bn()
elif args.type == 1:
    model = MobileNetV2withConvInsert1_bn()
elif args.type == 2:
    model = MobileNetV2withConvInsert2_bn()
elif args.type == 3:
    model = MobileNetV2withConvInsert3_bn()
elif args.type == 4:
    model = MobileNetV2withConvInsert4_bn()
elif args.type == 5:
    model = MobileNetV2withConvInsert5_bn()
elif args.type == 6:
    model = MobileNetV2withConvInsert6_bn()
elif args.type == 7:
    model = MobileNetV2withConvInsert7_bn()
elif args.type < 0:
    # origin
    model = MobileNetOrigin()
if args.type < 0:
    input1 = torch.rand(1,3,224,224)
    input2 = torch.rand(1,32,112,112)
    input3 = torch.rand(1,1280,7,7)
    macs,params = profile(model.mobilenetv2_part1,inputs = (input1,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
    macs,params = profile(model.mobilenetv2_part2,inputs = (input2,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
    part3 = nn.Sequential(*[model.reshape,model.mobilenetv2_part3])
    macs,params = profile(part3,inputs = (input3,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
if args.type == 3:
    input1 = torch.rand(1,3,224,224)
    input2 = torch.rand(1,16,28,28)
    input3 = torch.rand(1,220,7,7)
    part1 = nn.Sequential(*[model.mobilenetv2_part1,model.conv1])
    part2 = nn.Sequential(*[model.t_conv1,model.mobilenetv2_part2,model.conv2])
    part3 = nn.Sequential(*[model.t_conv2,model.reshape,model.mobilenetv2_part3])
    macs,params = profile(part1,inputs = (input1,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
    macs,params = profile(part2,inputs = (input2,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
    macs,params = profile(part3,inputs = (input3,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
if args.type == 7:
    input1 = torch.rand(1,3,224,224)
    input2 = torch.rand(1,8,14,14)
    input3 = torch.rand(1,30,7,7)
    part1 = nn.Sequential(*[model.mobilenetv2_part1,model.conv1,model.conv2])
    part2 = nn.Sequential(*[model.t_conv2,model.t_conv1,model.mobilenetv2_part2,model.conv3,model.conv4])
    part3 = nn.Sequential(*[model.t_conv4,model.t_conv3,model.reshape,model.mobilenetv2_part3])
    macs,params = profile(part1,inputs = (input1,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
    macs,params = profile(part2,inputs = (input2,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
    macs,params = profile(part3,inputs = (input3,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)