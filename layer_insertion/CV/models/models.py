import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class MobileNetV2withConvInsert0_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert0_bn, self).__init__()
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
        self.conv2 = nn.Conv2d(1280, 320, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(320, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier
        self.convsets = [self.conv1, self.t_conv1, self.conv2, self.t_conv2]

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
        self.convsets = [self.conv1, self.t_conv1, self.conv2, self.t_conv2]

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


class MobileNetV2withConvInsert2_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert2_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 8, (4, 4), (4, 4))
        self.t_conv1 = nn.ConvTranspose2d(8, 32, (4, 4), (4, 4))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv2 = nn.Conv2d(1280, 100, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(100, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier
        self.convsets = [self.conv1, self.t_conv1, self.conv2, self.t_conv2]

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


class MobileNetV2withConvInsert3_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert3_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 16, (4, 4), (4, 4))
        self.t_conv1 = nn.ConvTranspose2d(16, 32, (4, 4), (4, 4))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv2 = nn.Conv2d(1280, 220, (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(220, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier
        self.convsets = [self.conv1, self.t_conv1, self.conv2, self.t_conv2]

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


class MobileNetV2withConvInsert4_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert4_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 32, (4, 4), (4, 4))
        self.conv2 = nn.Conv2d(32, 16, (1, 1), (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(16, 32, (1, 1), (1, 1))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv3 = nn.Conv2d(1280, 400, (1, 1))
        self.conv4 = nn.Conv2d(400, 180, (1, 1))
        self.t_conv4 = nn.ConvTranspose2d(180, 400, (1, 1))
        self.t_conv3 = nn.ConvTranspose2d(400, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier
        self.convsets = [
            self.conv1,
            self.t_conv1,
            self.conv2,
            self.t_conv2,
            self.conv3,
            self.t_conv3,
            self.conv4,
            self.t_conv4,
        ]

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv3(output)
        output = self.conv4(output)
        output = self.t_conv4(output)
        output = self.t_conv3(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


class MobileNetV2withConvInsert5_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert5_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 32, (4, 4), (4, 4))
        self.conv2 = nn.Conv2d(32, 4, (1, 1), (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(4, 32, (1, 1), (1, 1))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv3 = nn.Conv2d(1280, 400, (1, 1))
        self.conv4 = nn.Conv2d(400, 100, (1, 1))
        self.t_conv4 = nn.ConvTranspose2d(100, 400, (1, 1))
        self.t_conv3 = nn.ConvTranspose2d(400, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier
        self.convsets = [
            self.conv1,
            self.t_conv1,
            self.conv2,
            self.t_conv2,
            self.conv3,
            self.t_conv3,
            self.conv4,
            self.t_conv4,
        ]

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv3(output)
        output = self.conv4(output)
        output = self.t_conv4(output)
        output = self.t_conv3(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


class MobileNetV2withConvInsert6_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert6_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 32, (8, 8), (8, 8))
        self.conv2 = nn.Conv2d(32, 16, (1, 1), (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(16, 32, (1, 1), (1, 1))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (8, 8), (8, 8))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv3 = nn.Conv2d(1280, 300, (1, 1))
        self.conv4 = nn.Conv2d(300, 50, (1, 1))
        self.t_conv4 = nn.ConvTranspose2d(50, 300, (1, 1))
        self.t_conv3 = nn.ConvTranspose2d(300, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier
        self.convsets = [
            self.conv1,
            self.t_conv1,
            self.conv2,
            self.t_conv2,
            self.conv3,
            self.t_conv3,
            self.conv4,
            self.t_conv4,
        ]

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv3(output)
        output = self.conv4(output)
        output = self.t_conv4(output)
        output = self.t_conv3(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output


class MobileNetV2withConvInsert7_bn(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV2withConvInsert7_bn, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        self.mobilenetv2_part1 = nn.Sequential(*[conv, bn])
        self.conv1 = nn.Conv2d(32, 32, (8, 8), (8, 8))
        self.conv2 = nn.Conv2d(32, 8, (1, 1), (1, 1))
        self.t_conv2 = nn.ConvTranspose2d(8, 32, (1, 1), (1, 1))
        self.t_conv1 = nn.ConvTranspose2d(32, 32, (8, 8), (8, 8))
        self.mobilenetv2_part2 = nn.Sequential(
            *[nn.ReLU6(inplace=False), model.features[1:]]
        )
        self.conv3 = nn.Conv2d(1280, 300, (1, 1))
        self.conv4 = nn.Conv2d(300, 30, (1, 1))
        self.t_conv4 = nn.ConvTranspose2d(30, 300, (1, 1))
        self.t_conv3 = nn.ConvTranspose2d(300, 1280, (1, 1))
        self.reshape = Reshape1()
        self.mobilenetv2_part3 = models.mobilenet_v2(pretrained=True).classifier
        self.convsets = [
            self.conv1,
            self.t_conv1,
            self.conv2,
            self.t_conv2,
            self.conv3,
            self.t_conv3,
            self.conv4,
            self.t_conv4,
        ]

    def forward(self, input):
        output = self.mobilenetv2_part1(input)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.t_conv2(output)
        output = self.t_conv1(output)
        output = self.mobilenetv2_part2(output)

        output = self.conv3(output)
        output = self.conv4(output)
        output = self.t_conv4(output)
        output = self.t_conv3(output)
        output = self.reshape(output)
        output = self.mobilenetv2_part3(output)
        return output
