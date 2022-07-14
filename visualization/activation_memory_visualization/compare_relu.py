'''
Author: Beta Cat 466904389@qq.com
Date: 2022-06-14 01:24:39
LastEditors: Beta Cat 466904389@qq.com
LastEditTime: 2022-06-28 23:49:38
FilePath: /research/gpipe_test/visualization/activation_memory_visualization/compare_relu.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import seaborn as sns

# model
model = models.mobilenet_v2(pretrained=True)
part1 = model.features[0]
part2 = model.features[0:]

feature = model.features[0].children()
conv = next(feature)
bn = next(feature)
part3 = nn.Sequential(*[conv, bn])
feature = model.features[-1].children()
conv = next(feature)
bn = next(feature)
part4 = nn.Sequential(*[model.features[0:-1], conv, bn])
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
trainset = torchvision.datasets.CIFAR10(
    root="../../../data", train=True, download=True, transform=transform_train
)
image = trainset[0][0]
image = image.view(1, 3, 224, 224)
output1 = part1(image)
output1 = output1.view(-1).detach().numpy()
output2 = part2(image)
output2 = output2.view(-1).detach().numpy()
output3 = part3(image)
output3 = output3.view(-1).detach().numpy()
output4 = part4(image)
output4 = output4.view(-1).detach().numpy()
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# print(output1.size)
plt.subplot(2, 2, 1)

data = sns.histplot(output1,bins = 100)
plt.title("Activation Value Distribution(conv+bn+relu)")
plt.xlabel("(a) Activation Value")
plt.ylabel("Count of Values")

plt.subplot(2, 2, 2)
data = sns.histplot(output3,bins = 100)
plt.title("Activation Value Distribution(conv+bn)")
plt.xlabel("(b) Activation Value")
plt.ylabel("Count of Values")


plt.subplot(2, 2, 3)
hist, bin_edge = np.histogram(output1)
cdf = np.cumsum(hist/sum(hist))
plt.xlabel("(c) Activation Value")
plt.ylabel("Cummulative Distribution")
plt.title("Activation CDF(conv+bn+relu)")
# plt.xlim([output1.min(),output1.max()])
plt.plot(bin_edge[1:], cdf, label="CDF")
# plt.legend()
plt.subplot(2, 2, 4)
hist, bin_edge = np.histogram(output3)
cdf = np.cumsum(hist/sum(hist))
plt.xlabel("(d) Activation Value")
plt.ylabel("Cummulative Distribution")
plt.title("Activation CDF(conv+bn)")
# plt.xlim([output3.min(),output3.max()])
plt.plot(bin_edge[1:], cdf, label="CDF")
# plt.hist(output2, bins=100, density=True)
# plt.title("distribution(sencondlast layer)")
# plt.xlabel("value")
# plt.ylabel("number of values")

# plt.subplot(2, 2, 4)
# plt.hist(output4, bins=100, density=True)
# plt.title("distribution(without relu)")
# plt.xlabel("value")
# plt.ylabel("number of values")
# plt.legend()
plt.tight_layout()
plt.show()
