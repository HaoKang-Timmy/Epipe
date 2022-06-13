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

plt.subplot(2, 2, 1)

<<<<<<< Updated upstream
data = sns.histplot(output1, bins=100)
plt.title("activation memory distribution(conv+bn+relu)")
plt.xlabel("(a) activation memory value")
plt.ylabel("count of values")

plt.subplot(2, 2, 2)
data = sns.histplot(output3, bins=100)
plt.title("activation memory distribution(conv+bn)")
plt.xlabel("(b) activation memory value")
plt.ylabel("count of values")
=======
data = sns.histplot(output1,bins = 100)
plt.title("Activation Memory Distribution(conv+bn+relu)")
plt.xlabel("(a) Activation Memory Value")
plt.ylabel("Count of Values")

plt.subplot(2, 2, 2)
data = sns.histplot(output3,bins = 100)
plt.title("Activation Memory Distribution(conv+bn)")
plt.xlabel("(b) Activation Memory Value")
plt.ylabel("Count of Values")
>>>>>>> Stashed changes


plt.subplot(2, 2, 3)
hist, bin_edge = np.histogram(output1)
<<<<<<< Updated upstream
cdf = np.cumsum(hist / sum(hist))
plt.xlabel("(c) weight value")
plt.ylabel("cummulative distribution")
plt.title("activation memory CDF(conv+bn+relu)")
=======
cdf = np.cumsum(hist/sum(hist))
plt.xlabel("(c) Activation Memory Value")
plt.ylabel("Cummulative Distribution")
plt.title("Activation Memory CDF(conv+bn+relu)")
>>>>>>> Stashed changes
# plt.xlim([output1.min(),output1.max()])
plt.plot(bin_edge[1:], cdf, label="CDF")
plt.legend()
plt.subplot(2, 2, 4)
hist, bin_edge = np.histogram(output3)
<<<<<<< Updated upstream
cdf = np.cumsum(hist / sum(hist))
plt.xlabel("(d) weight value")
plt.ylabel("cummulative distribution")
plt.title("activation memory CDF(conv+bn)")
=======
cdf = np.cumsum(hist/sum(hist))
plt.xlabel("(d) Activation Memory Value")
plt.ylabel("Cummulative Distribution")
plt.title("Activation Memory CDF(conv+bn)")
>>>>>>> Stashed changes
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
plt.legend()
plt.tight_layout()
plt.show()
