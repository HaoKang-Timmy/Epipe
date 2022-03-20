import torch.nn as nn
import torch
class nlp_sequential(nn.Module):
    def __init__(self,layers:list):
        super(nlp_sequential,self).__init__()
        self.layers = layers[0]
    def forward(self, output:torch.tensor,mask:torch.tensor):
        for layer in self.layers:
            output = layer(output,mask)
            output = output[0]
        return output

