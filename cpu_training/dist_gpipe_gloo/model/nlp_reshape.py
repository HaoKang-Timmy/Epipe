import torch.nn as nn
import torch


class nlp_sequential(nn.Module):
    def __init__(self, layers: list):
        super(nlp_sequential, self).__init__()
        self.layers = layers[0]

    def forward(self, output: torch.tensor, mask: torch.tensor):
        for i, layer in enumerate(self.layers):
            output = layer(output, mask)
            output = output[0]
        return output


class combine_embeding(nn.Module):
    def __init__(self, layers: list, embed_layer):
        super(combine_embeding, self).__init__()
        self.layers = layers[0]
        self.embed_layer = embed_layer[0]

    def forward(self, input: torch.tensor, mask: torch.tensor):
        output = self.embed_layer(input)

        output = self.layers(output, mask)
        output = output
        return output


class combine_classifier(nn.Module):
    def __init__(self, layers: list, classifier):
        super(combine_classifier, self).__init__()
        self.layers = layers[0]
        self.classifier = classifier[0]

    def forward(self, output: torch.tensor, mask: torch.tensor):
        # for i, layer in enumerate(self.layers):
        output = self.layers(output, mask)
        output = output
        output = self.classifier(output)
        return output
