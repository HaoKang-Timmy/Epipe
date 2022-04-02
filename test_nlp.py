# from model import resnet
import torchvision.models as models
from dist_gpipe import dist_gpipe_nlp,Reshape1,Reshape2,nlp_sequential
# from model import MobileNetV2, ResNet18
import model as md
import torch.nn as nn
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from dist_gpipe.distributedlayers.distributed_layers import ForwardSendLayers,ForwardReceiveLayers
import torchvision
import torch.distributed as dist
import argparse
# def foo(rank):
#     print("rank",rank)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--chunks',default = 4, type = int)
parser.add_argument('--log-dir',default = './my_gpipe', type = str)
parser.add_argument('--pretrained',default = 0,action= 'store_true')
parser.add_argument('--warmup',default = 0,action= 'store_true')
parser.add_argument('--lr',default = 2e-5, type = float)
parser.add_argument('--epoches',default = 80, type = int)
parser.add_argument('--batches',default = 16, type = int)
parser.add_argument('--quant',default = 0, type = int)
parser.add_argument('--prun',default = 0.0, type = float)
parser.add_argument('--Adam',default = 0,action= 'store_true')
def main():
    args = parser.parse_args()
    # if args.pretrained:
    #     model = models.mobilenet_v2(pretrained=True)
    #     print("use pretrained model")
    #     model.classifier[-1] = torch.nn.Linear(1280,10)
    #     layer1 = [model.features[0]]
    #     layer2 = [model.features[1:3]]
    #     layer3 = [model.features[3:7]]
    #     layer4 = [model.features[7:]]
    #     layer5 = [Reshape1(),model.classifier]
    # else:
    #     model = md.MobileNetV2()
    #     layer1 = [model.conv1,model.bn1]
    #     layer2 = [model.layers[0:3]]
    #     layer3 = [model.layers[3:7]]
    #     layer4 = [model.layers[7:],model.conv2]
    #     layer5 = [model.bn2,Reshape1(), model.linear]
    
    #     print("use kuangliu model")
    # print(model)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    from datasets import load_dataset
    train_dataset = load_dataset("glue","rte",split='train')
    val_dataset = load_dataset("glue","rte",split='validation')
    def encode(examples):
        return tokenizer(examples['sentence1'],examples['sentence2'], truncation=True, padding="max_length",max_length = 128)
    train_dataset = train_dataset.map(encode, batched=True)
    # print(train_dataset)
    val_dataset = val_dataset.map(encode, batched=True)
    val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    import torch
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset.set_format(type='torch', columns=['input_ids','labels','attention_mask'])
    val_dataset.set_format(type='torch', columns=['input_ids','labels','attention_mask'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16,num_workers = 12, pin_memory = True, drop_last = True,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16,num_workers = 12, pin_memory = True, drop_last = True,shuffle=False)
    layer1 = model.roberta.embeddings
    layer2 = nlp_sequential([model.roberta.encoder.layer[0:3]])
    layer3 = nlp_sequential([model.roberta.encoder.layer[3:7]])
    layer4 = nlp_sequential([model.roberta.encoder.layer[7:]])
    layer5 = model.classifier
    attention_mask = torch.rand([1,1,1,128])
    input = torch.rand([1,128]).type(torch.int)
    output = layer1(input)

    output = layer2(output,attention_mask)
    print("over")
    partition = [layer1,layer2,layer3,layer4,layer5]
    criterion = nn.CrossEntropyLoss()

    # if args.pretrained:
    #     transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    #     transform_test = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    #     input_size = [args.batches,3,224,224]
    # else:
    #     transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])

    #     transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    #     input_size = [args.batches,3,32,32]
    # trainset = torchvision.datasets.CIFAR10(
    # root='./data', train=True, download=True, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(
    # trainset, batch_size=args.batches, shuffle=True, num_workers=12,drop_last = True)

    # testset = torchvision.datasets.CIFAR10(
    # root='./data', train=False, download=True, transform=transform_test)
    # val_loader = torch.utils.data.DataLoader(
    # testset, batch_size=args.batches, shuffle=False, num_workers=12,drop_last = True)

    settings = {}
    settings['lr'] = args.lr
    settings['wd'] = 1e-6
    settings['momentum'] = 0.9
    settings['quantization'] = args.quant
    print(args.warmup)
    settings['warmup'] = args.warmup

    settings['prun'] = args.prun
    settings['quant'] = args.quant
    settings['Adam'] = args.Adam
    model  = dist_gpipe_nlp(partition,[0,1,2,3,0],args.chunks,input_size=[16,128],criterion = criterion,save_path = args.log_dir,settings = settings)

    # len_train_loader,len_val_loader = len(train_loader),len(val_loader)
    
    model.session(train_dataloader,val_dataloader,args.epoches,settings)
    
if __name__ == '__main__':
    main()

