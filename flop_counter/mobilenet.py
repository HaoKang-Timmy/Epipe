import torch
from thop import profile
import torchvision.models as models

model = models.vgg11()
input = torch.rand([64, 3, 224, 224])
mac1, param1 = profile(model, inputs=(input,))
print(mac1)
