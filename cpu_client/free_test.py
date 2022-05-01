import torch
import time
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=True)
model = model.features[0]
# inputs
input = torch.rand([32, 3, 224, 224])
start = time.time()
input = model(input)
print(time.time() - start)
input = input.view(-1)
start = time.time()
medium = torch.median(input)
print(time.time() - start)
start = time.time()
print(input.shape[0])
medium = torch.kthvalue(input, (input.shape[0]))
print(time.time() - start)
start = time.time()
src, index = torch.sort(input)
print(time.time() - start)
