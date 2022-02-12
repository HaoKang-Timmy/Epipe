import torch.nn as nn
class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self,input):
        return input.view(input.size(0),-1)