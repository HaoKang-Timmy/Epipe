import torch
from functions import *
import time
import torch.nn as nn
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description="CPU test for nlp and cv")
parser.add_argument("--tasktype", default="cv", type=str)
parser.add_argument("--layer", default="first", type=str)
args = parser.parse_args()


class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


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


split_bits = 2
if args.tasktype == "cv":
    if args.layer == "last":
        model = mobilenet_v2(pretrained=True)
        model1 = [Reshape1(), model.classifier]
        model1 = nn.Sequential(*model1)
    else:
        model = mobilenet_v2(pretrained=True)
        model1 = [model.classifier[0:1]]
        model1 = nn.Sequential(*model1)
else:
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    if args.layer == "first":
        part1 = [model.roberta.embeddings]
        part2 = nlp_sequential([model.roberta.encoder.layer[0:1]])
        model1 = combine_embeding([part2], part1)
    else:
        part1 = nlp_sequential([model.roberta.encoder.layer[-1:]])
        part2 = model.classifier
        model1 = combine_classifier([part1], [part2])
min_step = torch.rand([2**split_bits, 2])
firstlayer_time = []
sq_time = []
sdq_time = []
pca_time = []
de_pca_time = []
i_list = []
for i in range(32):
    i = i + 1
    print(i)
    i_list.append(i)
    if args.tasktype == "nlp":
        if args.layer == "first":
            input = torch.rand([i, 128, 768]).requires_grad_()
            some = torch.rand([i, 128, 768])
            mask = torch.rand([i, 1, 1, 128])
            model_input = torch.randint(0, 10000, (i, 128)).type(torch.long)
        else:
            mask = torch.rand([i, 1, 1, 128])
            input = torch.rand([i, 128, 768]).requires_grad_()
            some = torch.rand([i, 128, 768])
            model_input = input
        start = time.time()
        output = model1(model_input, mask)
        end = time.time() - start
    else:
        if args.layer == "first":
            input = torch.rand([i, 32, 112, 112]).requires_grad_()
            some = torch.rand([i, 32, 112, 112])
        else:
            input = torch.rand([i, 1280, 7, 7]).requires_grad_()
            some = torch.rand([i, 1280, 7, 7])
        start = time.time()
        output = model1(input)
        end = time.time() - start
    firstlayer_time.append(end)
    with torch.no_grad():

        # # start = time.time()
        # min_step,output = SortQuantization(input,6,2,min_step)
        # # print(time.time() - start)
        # min_step1 = torch.rand([2**split_bits, 2])
        start = time.time()
        # print(input)
        new_min_step, new_output = FastQuantization(input, 6, 2, min_step)

        end = time.time() - start
        sq_time.append(end)
        some = some.view(-1)
        new_output = new_output.view(-1)
        start = time.time()
        dequant = FastDeQuantization(new_output, 6, 2, new_min_step, some)
        end = time.time() - start
        sdq_time.append(end)
        start = time.time()
        U, S, V = torch.svd_lowrank(input, q=2)
        end = time.time() - start
        pca_time.append(end)
        start = time.time()
        V = V.transpose(-1, -2)
        S = torch.diag_embed(S)
        pca = torch.matmul(U[..., :, :], S[..., :, :])
        new = torch.matmul(pca[..., :, :], V[..., :, :])
        end = time.time() - start
        de_pca_time.append(end)
l1 = plt.plot(i_list, sq_time, label="sortquant", marker="o")
l2 = plt.plot(i_list, sdq_time, label="sortdequant", marker="o")
l3 = plt.plot(i_list, pca_time, label="pcar2", marker="o")
l4 = plt.plot(i_list, de_pca_time, label="pca decode", marker="o")
l5 = plt.plot(i_list, de_pca_time, label="first layer", marker="o")
plt.title("CIFAR Last Layer")
plt.xlabel("batch size")
plt.ylabel("execution time")
plt.legend()
plt.savefig("./cifar_last.jpg")
