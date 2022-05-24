import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


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
        # output = output + input
        for i, layer in enumerate(self.others):
            output = layer(output, mask)
            output = output[0]
        return output


class NLPSequential(nn.Module):
    def __init__(self, layers: list):
        super(NLPSequential, self).__init__()
        self.layers = layers[0]

    def forward(self, output: torch.tensor, mask: torch.tensor):
        for i, layer in enumerate(self.layers):
            output = layer(output, mask)
            output = output[0]
        return output


class Robertabase(nn.Module):
    def __init__(self, model=None) -> None:
        super(Robertabase, self).__init__()
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        embedding = model.roberta.embeddings
        attention = model.roberta.encoder.layer[0].attention
        medium = model.roberta.encoder.layer[0].intermediate
        output_layer = model.roberta.encoder.layer[0].output
        roberta_layers = model.roberta.encoder.layer[1:]
        self.part1 = EmbeddingAndAttention([embedding], [attention])
        self.part2 = CombineLayer([medium], [output_layer], [roberta_layers])
        self.part3 = model.classifier

    def forward(self, input, mask):
        mask = torch.reshape(mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])])
        mask = (1.0 - mask) * -1e9
        output = self.part1(input, mask)
        output = self.part2(output, mask)
        output = self.part3(output)
        return output


class RobertabaseLinear(nn.Module):
    def __init__(self, model=None, linear_channel=768) -> None:
        super(RobertabaseLinear, self).__init__()
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        embedding = model.roberta.embeddings
        attention = model.roberta.encoder.layer[0].attention
        medium = model.roberta.encoder.layer[0].intermediate
        output_layer = model.roberta.encoder.layer[0].output
        roberta_layers = model.roberta.encoder.layer[1:]
        self.part1 = EmbeddingAndAttention([embedding], [attention])
        self.part2 = CombineLayer([medium], [output_layer], [roberta_layers])
        self.part3 = model.classifier
        self.linear1 = nn.Linear(768, linear_channel)
        self.linear2 = nn.Linear(linear_channel, 768)

    def forward(self, input, mask):
        mask = torch.reshape(mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])])
        mask = (1.0 - mask) * -1e9
        output = self.part1(input, mask)
        output = self.linear1(output)
        output = self.part2(output, mask)
        output = self.linear2(output)
        output = self.part3(output)
        return output


class RobertabaseLinear1(nn.Module):
    def __init__(self, model=None, linear_channel=768, eye=None) -> None:
        super(RobertabaseLinear1, self).__init__()
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        embedding = model.roberta.embeddings
        self.embedding = embedding
        self.part1 = model.roberta.encoder.layer[0]
        self.part2 = NLPSequential([model.roberta.encoder.layer[1:]])
        self.part3 = model.classifier
        # self.matrix = torch.nn.Parameter(torch.eye(768))
        if eye is None:
            self.matrix1 = torch.nn.Parameter(torch.eye(768))
            self.linear1 = nn.Linear(768, linear_channel)
            self.linear2 = nn.Linear(linear_channel, 768)
        else:
            self.matrix1 = torch.nn.Parameter(torch.eye(768))
            self.matrix2 = torch.nn.Parameter(torch.eye(768))
        self.eye = eye

    def forward(self, input, mask):
        mask = torch.reshape(mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])])
        mask = (1.0 - mask) * -1e9
        input = self.embedding(input)
        output = self.part1(input, mask)
        output = output[0]
        output = output @ self.matrix1
        output = self.part2(output, mask)
        if self.eye is None:
            output = self.linear1(output)
            output = self.linear2(output)
        else:
            output = output @ self.matrix2
        output = self.part3(output)
        return output


class RobertabaseLinear2(nn.Module):
    def __init__(self, model=None, linear_channel=768, eye=None) -> None:
        super(RobertabaseLinear2, self).__init__()
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        embedding = model.roberta.embeddings
        self.embedding = embedding
        self.part1 = model.roberta.encoder.layer[0]
        self.part2 = NLPSequential([model.roberta.encoder.layer[1:-1]])
        self.part3 = NLPSequential([model.roberta.encoder.layer[-1:]])
        self.part4 = model.classifier
        # self.matrix = torch.nn.Parameter(torch.eye(768))
        if eye is None:
            self.matrix1 = torch.nn.Parameter(torch.eye(768))
            self.linear1 = nn.Linear(768, linear_channel)
            self.linear2 = nn.Linear(linear_channel, 768)
        else:
            self.matrix1 = torch.nn.Parameter(torch.eye(768))
            self.matrix2 = torch.nn.Parameter(torch.eye(768))
        self.eye = eye

    def forward(self, input, mask):
        mask = torch.reshape(mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])])
        mask = (1.0 - mask) * -1e9
        input = self.embedding(input)
        output = self.part1(input, mask)
        output = output[0]
        output = output @ self.matrix1
        output = self.part2(output, mask)
        if self.eye is None:
            output = self.linear1(output)
            output = self.linear2(output)
        else:
            output = output @ self.matrix2
        output = self.part3(output, mask)
        output = self.part4(output)
        return output
