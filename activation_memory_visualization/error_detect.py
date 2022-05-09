'''
Author: Beta Cat 466904389@qq.com
Date: 2022-05-09 18:10:32
LastEditors: Beta Cat 466904389@qq.com
LastEditTime: 2022-05-09 22:46:58
FilePath: /research/gpipe_test/activation_memory_visualization/mobilenet_firstlayer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from utils import Fakequantize,SortQuantization,abl_err,relative_err
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision

# model
model = models.mobilenet_v2(pretrained= True)
part1 = model.features[0:1]
transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
)
trainset = torchvision.datasets.CIFAR10(
    root="../../data", train=True, download=True, transform=transform_train
)
image = trainset[0][0]
image = image.view(1,3,224,224)
output1 = part1(image)
label = output1.clone().detach()
quant = Fakequantize.apply(output1,8)
squant = SortQuantization.apply(output1,6,2)
# print(output1)
# print(squant)
# print(quant - label)
# print(label)
print(relative_err(quant,label))
print(relative_err(squant,label))

