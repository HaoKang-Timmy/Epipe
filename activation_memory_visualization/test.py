'''
Author: Beta Cat 466904389@qq.com
Date: 2022-05-09 19:28:28
LastEditors: Beta Cat 466904389@qq.com
LastEditTime: 2022-05-09 21:25:34
FilePath: /research/gpipe_test/activation_memory_visualization/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
# model
model = models.mobilenet_v2(pretrained= True)
part1 = model.features[0:2]
part2 = model.features[0:]

feature = model.features[0].children()
conv = next(feature)
bn = next(feature)
part3 = nn.Sequential(*[conv, bn])
feature = model.features[-1].children()
conv = next(feature)
bn = next(feature)
part4 = nn.Sequential(*[model.features[0:-1],conv,bn])
# part3 = model.features[0:2]
# part4 = model.features[0:-1]
# transform_train = transforms.Compose(
#         [
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ]
# )
# trainset = torchvision.datasets.CIFAR10(
#     root="../../data", train=True, download=True, transform=transform_train
# )
