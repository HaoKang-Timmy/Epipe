import torch
import torch.nn as nn
class EmbeddingAndAttention(nn.Module):
    def __init__(self, embedding_layer, attention_layer):
        super(EmbeddingAndAttention, self).__init__()
        self.embedding_layer = embedding_layer[0]
        self.attention_layer = attention_layer[0]

    def forward(self, input, mask):
        output = self.embedding_layer(input)
        output = self.attention_layer(output, mask)
        return output[0]


class CombineLayer(nn.Module):
    def __init__(self, mediumlayer, output_layer, others):
        super(CombineLayer, self).__init__()
        self.medium = mediumlayer[0]
        self.outputlayer = output_layer[0]
        self.others = others[0]

    def forward(self, input, mask):
        output = self.medium(input)
        output = self.outputlayer(output, input)
        for i, layer in enumerate(self.others):
            output = layer(output, mask)
            output = output[0]
        return output
class nlp_sequential(nn.Module):
    def __init__(self, layers: list):
        super(nlp_sequential, self).__init__()
        self.layers = layers[0]

    def forward(self, output: torch.tensor, mask: torch.tensor):
        for i, layer in enumerate(self.layers):
            output = layer(output, mask)
            output = output[0]
        return output