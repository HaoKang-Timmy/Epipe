import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import torchvision.models as models
import torch.nn.functional as F
from .utils import *


class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


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


# condition
class RobertabaseLinear1(nn.Module):
    def __init__(self, model=None, linear_channel=768, eye=None) -> None:
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
    def __init__(self, model=None, linear_channel=768, eye=None) -> None:
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
    def __init__(self, model=None, linear_channel=768, eye=None) -> None:
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

        output = output @ self.matrix1
        output = self.part2(output, mask)
        if self.eye is None:
            output = self.linear1(output)
            output = self.linear2(output)
        else:
            output = output @ self.matrix2
        output = self.part3(output)
        return output


class RobertabaseLinear4(nn.Module):
    def __init__(self, model=None, linear_channel=768, eye=None) -> None:
        super(RobertabaseLinear4, self).__init__()
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


class Robertawithcompress(nn.Module):
    def __init__(self, args, device, batchsize):
        super(Robertawithcompress, self).__init__()
        self.args = args
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        embedding = model.roberta.embeddings
        attention = model.roberta.encoder.layer[0].attention
        medium = model.roberta.encoder.layer[0].intermediate
        output_layer = model.roberta.encoder.layer[0].output
        roberta_layers = model.roberta.encoder.layer[1:]
        self.part1 = EmbeddingAndAttention([embedding], [attention])
        self.part2 = CombineLayer([medium], [output_layer], [roberta_layers])
        self.part3 = model.classifier
        if args.linear != 0:
            self.linear1 = torch.nn.Linear(768, args.linear).to(device)
            self.linear2 = torch.nn.Linear(args.linear, 768).to(device)
            self.linear3 = torch.nn.Linear(768, args.linear).to(device)
            self.linear4 = torch.nn.Linear(args.linear, 768).to(device)
        if args.prun != 0:
            self.topk_layer1 = TopkLayer(args.prun)
            self.topk_layer2 = TopkLayer(args.prun)
        if args.powerpca != 0:
            self.powerpca_layer1 = PowerSVDLayerNLP(
                args.powerpca, (int(batchsize), 128, 768), args.poweriter, device
            )
            self.powerpca_layer2 = PowerSVDLayerNLP(
                args.powerpca, (int(batchsize), 128, 768), args.poweriter, device
            )

    def forward(self, input, mask):
        args = self.args
        mask = torch.reshape(mask, [int(mask.shape[0]), 1, 1, int(mask.shape[-1])])
        mask = (1.0 - mask) * -1e9
        outputs = self.part1(input, mask)
        if args.prun != 0:
            outputs = self.topk_layer1(outputs)
        if args.quant != 0:
            outputs = Fakequantize.apply(outputs, args.quant)
        if args.pca != 0:
            outputs = PCAQuantize.apply(outputs, args.pca)
        if args.linear != 0:
            outputs = self.linear1(outputs)
            outputs = self.linear2(outputs)
        if args.sortquant != 0:
            outputs = SortQuantization.apply(outputs, args.quant, args.prun, args.sort)
        if args.powerpca != 0:
            outputs = self.powerpca_layer1(outputs)
        outputs = self.part2(outputs, mask)
        if args.prun != 0:
            outputs = self.topk_layer2(outputs)
        if args.quant != 0:
            outputs = Fakequantize.apply(outputs, args.quant)
        if args.pca != 0:
            outputs = PCAQuantize.apply(outputs, args.pca)
        if args.linear != 0:
            outputs = self.linear3(outputs)
            outputs = self.linear4(outputs)
        if args.sortquant != 0:
            outputs = SortQuantization.apply(outputs, args.quant, args.prun, args.sort)
        if args.powerpca != 0:
            outputs = self.powerpca_layer2(outputs)
        outputs = self.part3(outputs)
        return outputs


class MobileNetV2Compress(nn.Module):
    def __init__(self, args, rank, shape1=None, shape2=None):
        super(MobileNetV2Compress,self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(1280, 10)
        feature = model.features[0].children()
        conv = next(feature)
        bn = next(feature)
        if args.secondlayer == 0:
            self.part1 = nn.Sequential(*[conv, bn])
            self.part2 = nn.Sequential(*[nn.ReLU6(inplace=False), model.features[1:]])
            self.part3 = nn.Sequential(*[Reshape1(), model.classifier])
        else:
            self.part1 = nn.Sequential(*[model.features[0:1]])
            self.part2 = nn.Sequential(*[model.features[1:]])
            self.part3 = nn.Sequential(*[Reshape1(), model.classifier])
        if args.conv1 != 0:
            self.conv2d = torch.nn.Conv2d(32, 32, (4, 4), (4, 4))
            self.conv_t = torch.nn.ConvTranspose2d(32, 32, (4, 4), (4, 4))
        if args.conv2 != 0:
            self.conv2d1 = torch.nn.Conv2d(1280, 320, (1, 1))
            self.conv_t1 = torch.nn.ConvTranspose2d(320, 1280, (1, 1))
        self.args = args
        self.bool = 0
        self.rank = rank
        if args.powerrank != 0:
            self.svd1 = PowerSVDLayer1(args.powerrank, list(shape1), args.poweriter).to(
                self.rank
            )
            self.svd2 = PowerSVDLayer1(args.powerrank1, list(shape2), args.poweriter).to(
                self.rank
            )

    def forward(self, input):
        outputs = self.part1(input)
        args = self.args
        if args.sortquant != 0:
            outputs = FastQuantization.apply(outputs, args.quant, args.split)
        elif args.conv1 != 0:
            outputs = self.conv2d(outputs)
            # outputs = conv2d1(outputs)
            # outputs = conv_t1(outputs)
            outputs = self.conv_t(outputs)
        # elif args.channelquant != 0:
        #     outputs = ChannelwiseQuantization.apply(outputs, args.channelquant)
        elif args.powerrank != 0:

            outputs = self.svd1(outputs)
        elif args.svd != 0:
            outputs = ReshapeSVD.apply(outputs, args.svd)
        outputs = self.part2(outputs)
        if args.sortquant != 0:
            outputs = FastQuantization.apply(outputs, args.quant, args.split)
        elif args.conv2 != 0:
            outputs = self.conv2d1(outputs)
            outputs = self.conv_t1(outputs)
        elif args.powerrank1 != 0:

            outputs = self.svd2(outputs)
        elif args.svd != 0:
            outputs = ReshapeSVD.apply(outputs, args.svd)
        outputs = self.part3(outputs)
        return outputs
