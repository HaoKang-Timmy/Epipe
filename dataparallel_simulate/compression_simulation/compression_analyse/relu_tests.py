from re import sub
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.models import mobilenet_v2
from utils import FakeQuantize
from ignite.contrib.metrics.regression import mean_absolute_relative_error
import torch.nn as nn


def l2_disteance(input, label):
    difference = input - label
    difference = torch.abs(difference)
    return difference.mean()


transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="../../../data", train=True, download=True, transform=transform_train
)
sub_model = mobilenet_v2(pretrained=True).features[0]
children = sub_model.children()
conv = next(children)
bn = next(children)
model_withoutrelu = nn.Sequential(*[conv, bn])
print(conv)
output = sub_model(trainset[0][0].view(1, 3, 224, 224))
quantized_output = FakeQuantize.apply(output, 8)
# metric = mean_absolute_relative_error.MeanAbsoluteRelativeError()
print(l2_disteance(output, quantized_output))

output_withoutrelu = model_withoutrelu(trainset[0][0].view(1, 3, 224, 224))
quantized_output_withoutrelu = FakeQuantize.apply(output_withoutrelu, 8)
print(l2_disteance(output_withoutrelu, quantized_output_withoutrelu))
