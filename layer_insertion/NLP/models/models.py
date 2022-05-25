from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
from utils import *


class Robertapretrain(nn.Module):
    def __init__(self, args) -> None:
        super(Robertapretrain, self).__init__()
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        for param in model.parameters():
            param.require_grad = False
        embedding = model.roberta.embeddings
        attention = model.roberta.encoder.layer[0].attention
        medium = model.roberta.encoder.layer[0].intermediate
        output_layer = model.roberta.encoder.layer[0].output
        roberta_layers = model.roberta.encoder.layer[1:]
        self.part1 = EmbeddingAndAttention([embedding], [attention])
        self.part2 = CombineLayer([medium], [output_layer], [roberta_layers])
        # self.part3 = model.classifier
        if args.compressdim == -1:
            self.linear1 = nn.Linear(768, args.rank)
            self.linear2 = nn.Linear(args.rank, 768)
            self.linear3 = nn.Linear(768, args.rank)
            self.linear4 = nn.Linear(args.rank, 768)
        else:
            self.linear1 = nn.Linear(128, args.rank)
            self.linear2 = nn.Linear(args.rank, 128)
            self.linear3 = nn.Linear(128, args.rank)
            self.linear4 = nn.Linear(args.rank, 128)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.args = args

    def forward(self, input, mask):
        mask = torch.reshape(
            mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])]
        ).type(torch.float32)
        mask = (1.0 - mask) * -1e9
        label1 = self.part1(input, mask)
        if self.args.compressdim != -1:
            input1 = label1.permute((0, 2, 1))
            outputs = self.linear1(input1)
        else:
            outputs = self.linear1(label1)
        print(outputs.shape)
        outputs1 = self.linear2(outputs)
        if self.args.compressdim != -1:
            outputs1 = outputs1.permute((0, 2, 1))
        label2 = self.part2(label1, mask)
        if self.args.compressdim != -1:
            input2 = label2.permute((0, 2, 1))
            outputs = self.linear3(input2)
        else:
            outputs = self.linear3(label2)
        print(outputs.shape)
        outputs2 = self.linear4(outputs)
        if self.args.compressdim != -1:
            outputs2 = outputs2.permute((0, 2, 1))
        loss1 = self.criterion1(outputs1, label1)
        loss2 = self.criterion2(outputs2, label2)
        return loss1, loss2


class RobertabaseLinear1(nn.Module):
    def __init__(
        self, model=None, linear_channel=768, eye=None, compressdim=-1
    ) -> None:
        super(RobertabaseLinear1, self).__init__()
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


class RobertabaseLinear2(nn.Module):
    def __init__(
        self, model=None, linear_channel=768, eye=None, compressdim=-1
    ) -> None:
        super(RobertabaseLinear2, self).__init__()
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        embedding = model.roberta.embeddings
        attention = model.roberta.encoder.layer[0].attention
        medium = model.roberta.encoder.layer[0].intermediate
        output_layer = model.roberta.encoder.layer[0].output
        roberta_layers = model.roberta.encoder.layer[1:-1]
        self.part1 = EmbeddingAndAttention([embedding], [attention])
        self.part2 = CombineLayer([medium], [output_layer], [roberta_layers])
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

        output = self.part1(input, mask)
        # output = output[0]
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


class RobertabaseLinear3(nn.Module):
    def __init__(self, model=None, linear_channel=768, compressdim=-1) -> None:
        super(RobertabaseLinear3, self).__init__()
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
        # self.matrix = torch.nn.Parameter(torch.eye(768))
        if compressdim == -1:
            self.linear1 = torch.nn.Linear(768, linear_channel)
            self.linear2 = torch.nn.Linear(linear_channel, 768)
            self.linear3 = torch.nn.Linear(768, linear_channel)
            self.linear4 = torch.nn.Linear(linear_channel, 768)
        elif compressdim == -2:
            self.linear1 = torch.nn.Linear(128, linear_channel)
            self.linear2 = torch.nn.Linear(linear_channel, 128)
            self.linear3 = torch.nn.Linear(128, linear_channel)
            self.linear4 = torch.nn.Linear(linear_channel, 128)
        self.compressdim = compressdim

    def forward(self, input, mask):
        mask = torch.reshape(mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])])
        mask = (1.0 - mask) * -1e9
        output = self.part1(input, mask)

        if self.compressdim == -1:
            output = self.linear1(output)
            output = self.linear2(output)
        elif self.compressdim == -2:
            output = output.permute((0, 2, 1))
            output = self.linear1(output)
            output = self.linear2(output)
            output = output.permute((0, 2, 1))
        output = self.part2(output, mask)
        if self.compressdim == -1:
            output = self.linear3(output)
            output = self.linear4(output)
        elif self.compressdim == -2:
            output = output.permute((0, 2, 1))
            output = self.linear3(output)
            output = self.linear4(output)
            output = output.permute((0, 2, 1))
        output = self.part3(output)
        return output


class RobertabaseLinear4(nn.Module):
    def __init__(self, model=None, linear_channel=768, compressdim=-1) -> None:
        super(RobertabaseLinear4, self).__init__()
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        embedding = model.roberta.embeddings
        self.embedding = embedding
        self.part1 = model.roberta.encoder.layer[0]
        self.part2 = NLPSequential([model.roberta.encoder.layer[1:]])
        self.part3 = model.classifier
        # self.matrix = torch.nn.Parameter(torch.eye(768))
        print(compressdim)
        if compressdim == -1:
            self.linear1 = torch.nn.Linear(768, linear_channel)
            self.linear2 = torch.nn.Linear(linear_channel, 768)
            self.linear3 = torch.nn.Linear(768, linear_channel)
            self.linear4 = torch.nn.Linear(linear_channel, 768)
        elif compressdim == -2:
            self.linear1 = torch.nn.Linear(128, linear_channel)
            self.linear2 = torch.nn.Linear(linear_channel, 128)
            self.linear3 = torch.nn.Linear(128, linear_channel)
            self.linear4 = torch.nn.Linear(linear_channel, 128)
        self.compressdim = compressdim

    def forward(self, input, mask):
        mask = torch.reshape(mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])])
        mask = (1.0 - mask) * -1e9
        input = self.embedding(input)
        output = self.part1(input, mask)
        output = output[0]
        if self.compressdim == -1:
            output = self.linear1(output)
            output = self.linear2(output)
        elif self.compressdim == -2:
            output = output.permute((0, 2, 1))
            output = self.linear1(output)
            output = self.linear2(output)
            output = output.permute((0, 2, 1))
        output = self.part2(output, mask)
        if self.compressdim == -1:
            output = self.linear3(output)
            output = self.linear4(output)
        elif self.compressdim == -2:
            output = output.permute((0, 2, 1))
            output = self.linear3(output)
            output = self.linear4(output)
            output = output.permute((0, 2, 1))

        output = self.part3(output)
        return output


class RobertabaseLinearDecay(nn.Module):
    def __init__(self, model=None) -> None:
        super(RobertabaseLinearDecay, self).__init__()
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

        embedding = model.roberta.embeddings
        self.embedding = embedding
        self.part1 = model.roberta.encoder.layer[0]
        self.part2 = NLPSequential([model.roberta.encoder.layer[1:]])
        self.part3 = model.classifier
        self.matrix1 = torch.nn.Parameter(torch.eye(768))
        self.matrix2 = torch.nn.Parameter(torch.eye(768))
        self.matrix3 = torch.nn.Parameter(torch.eye(768))
        self.matrix4 = torch.nn.Parameter(torch.eye(768))
        self.step = 0

    def forward(self, input, mask):
        mask = torch.reshape(mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])])
        mask = (1.0 - mask) * -1e9
        input = self.embedding(input)
        output = self.part1(input, mask)
        output = output[0]
        output = output @ self.matrix1
        output = output @ self.matrix2
        output = self.part2(output, mask)
        output = output @ self.matrix3
        output = output @ self.matrix4
        output = self.part3(output)
        return output
