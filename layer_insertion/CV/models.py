import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class MobileNetV2withConvInsert(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert, self).__init__()
        self.mobilenetv2_part1 = models.mobilenet_v2(pretrained=True).features[0]
        self.conv1 = nn.Conv2d(32, 32, (2, 2), (2, 2))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (2, 2), (2, 2))
        self.mobilenetv2_part2 = models.mobilenet_v2(pretrained=True).features[1:]
        self.conv2 = nn.Conv2d(1280, 320, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(320, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


class MobileNetV2withConvInsert1(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert1, self).__init__()
        self.mobilenetv2_part1 = models.mobilenet_v2(pretrained=True).features[0]
        self.conv1 = nn.Conv2d(32, 32, (4, 4), (4, 4))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
        self.mobilenetv2_part2 = models.mobilenet_v2(pretrained=True).features[1:]
        self.conv2 = nn.Conv2d(1280, 360, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(360, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


class MobileNetV2withConvInsert2(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert2, self).__init__()
        self.mobilenetv2_part1 = models.mobilenet_v2(pretrained=True).features[0]
        self.conv1 = nn.Conv2d(32, 32, (2, 2), (2, 2))
        self.conv2 = nn.Conv2d(32, 32, (2, 2), (2, 2))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (2, 2), (2, 2))
        self.t_conv2 = nn.ConvTranspose2d(32, 32, (2, 2), (2, 2))
        self.mobilenetv2_part2 = models.mobilenet_v2(pretrained=True).features[1:]
        self.conv3 = nn.Conv2d(1280, 360, (1, 1))
        self.t_conv3 = nn.ConvTranspose2d(360, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.t_conv1(output)
        output = self.t_conv2(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv3(output)
        output = self.t_conv3(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


class MobileNetV2withConvInsert3(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert3, self).__init__()
        self.mobilenetv2_part1 = models.mobilenet_v2(pretrained=True).features[0]
        self.conv1 = nn.Conv2d(32, 20, (3, 3), (3, 3))
        self.t_conv1 = nn.ConvTranspose2d(20, 32, (3, 3), (3, 3))
        self.mobilenetv2_part2 = models.mobilenet_v2(pretrained=True).features[1:]
        self.conv2 = nn.Conv2d(1280, 360, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(360, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class MobileNetV2withConvInsert3_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert3_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 20, (3, 3), (3, 3))
        self.t_conv1 = nn.ConvTranspose2d(20, 32, (3, 3), (3, 3))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv2 = nn.Conv2d(1280, 360, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(360, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


class MobileNetV2withConvInsert1_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert1_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 32, (4, 4), (4, 4))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv2 = nn.Conv2d(1280, 240, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(240, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output
