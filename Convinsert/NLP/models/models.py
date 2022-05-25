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
