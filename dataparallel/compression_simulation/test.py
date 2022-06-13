from torchvision.models import mobilenet_v2, resnet34

model = mobilenet_v2()
print(model.classifier[-3])
