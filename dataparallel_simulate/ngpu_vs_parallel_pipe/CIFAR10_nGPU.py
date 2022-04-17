
import torchvision.models as models
import torch.nn as nn
import time
from transformers import get_scheduler
import torchvision.transforms as transforms
import torchvision
import torch
import datasets
import argparse
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
# from utils import accuracy,Reshape1,QuantizationLayer,DequantizationLayer,Fakequantize,TopkLayer,Topk_quantization,KMeansLayer
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
# parser.add_argument("--chunks", default=4, type=int)
parser.add_argument("--root", default="./data", type=str)
parser.add_argument("--log-dir", default="./test_cv.txt", type=str)
parser.add_argument("--pretrained", default=0, action="store_true")
parser.add_argument("--warmup", default=0, action="store_true")
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--batches", default=64, type=int)
parser.add_argument("--worker", default=4, type=int)
parser.add_argument("--quant", default=0, type=int)
parser.add_argument("--prun", default=0.0, type=float)
parser.add_argument("--avgpool", default=0, action="store_true")
parser.add_argument("--split", default=4, type=int)
parser.add_argument("--multi", default=0, action="store_true")
parser.add_argument("--kmeans", default=0, type = int)
parser.add_argument("--nocompress", default=0, action="store_true")
parser.add_argument("--dynamics", default=0, action="store_true")
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class Reshape1(nn.Module):
    def __init__(self):
        super(Reshape1, self).__init__()
        pass

    def forward(self, x):
        out = F.relu(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
def main():
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=args.worker,
                 args=(args.worker, args))
def main_worker(rank,process_num,args):
    dist.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:1235',
                                    world_size=args.worker, rank=rank)
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=args.root, train=True, download=True, transform=transform_train
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batches, shuffle=(train_sampler is None), num_workers=12, drop_last=True, sampler=train_sampler,pin_memory=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.root, train=False, download=True, transform=transform_test
    )
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batches, shuffle=False, num_workers=12, drop_last=True,pin_memory=True,
    )
#     pass
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[-1] = torch.nn.Linear(1280, 10)

    layer1 = nn.Sequential(*[model.features[0:1]])
    layer2 = nn.Sequential(*[model.features[1:]])
    layer3 = nn.Sequential(*[Reshape1(), model.classifier])
    # quant_layer1 = QuantizationLayer(args.quant)
    # dequant_layer1 = DequantizationLayer(args.quant)
    # quant_layer2 = QuantizationLayer(args.quant)
    # dequant_layer2 = DequantizationLayer(args.quant)
    layer1 = layer1.to(rank)
    layer2 = layer2.to(rank)
    layer3 = layer3.to(rank)

    # quant_layer1 = quant_layer1.to(rank)
    # dequant_layer1 = dequant_layer1.to(rank)
    # quant_layer2 =quant_layer2.to(rank)
    # dequant_layer2 = dequant_layer2.to(rank)
    layer1 = torch.nn.parallel.DistributedDataParallel(layer1)
    layer2 = torch.nn.parallel.DistributedDataParallel(layer2)
    layer3 = torch.nn.parallel.DistributedDataParallel(layer3)
    optimizer = torch.optim.SGD([{"params":layer1.parameters()},{"params":layer2.parameters()},{"params":layer3.parameters()}],lr = args.lr,momentum=0.9)

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=int(args.epochs/10), num_training_steps=args.epochs
    )

    criterion = nn.CrossEntropyLoss().to(rank)
# optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    for epoch in range(args.epochs):
        layer1.train()
        layer2.train()
        layer3.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        train_acc1 = 0.0
        time_avg = 0.0
        backward_avg = 0.0
        data_avg = 0.0
        start = time.time()
        for i, (image,label) in enumerate(train_loader):
            # start = time.time()
            image = image.to(rank,non_blocking = True)
            label = label.to(rank,non_blocking = True)
            datatime = time.time() - start


            
            outputs =layer1(image)

            outputs = layer2(outputs)

            # outputs,min,step = quant_layer2(outputs)
            # outputs = dequant_layer2(outputs,min,step,quant_layer2.backward_min,quant_layer2.backward_step)
            outputs = layer3(outputs)
            # print(outputs)
            # while(1):
            #     pass
            loss = criterion(outputs,label)
            acc,_ = accuracy(outputs,label,topk =(1,2))
            # pred = np.argmax(logits.cpu(), axis=1)
            # pred = torch.argmax(logits,dim = 1)
            # pred = np.argmax(logits.item(),axis = 1)
        
            # metric.add_batch(predictions = logits,references=batch['labels'])
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_acc1 += acc.item()
            
            end = time.time() - start
            backward_time = time.time() - backward_start
            backward_avg += backward_time
            data_avg += datatime 
            time_avg += end
            if i %20 == 0 and rank == 0:
                print("train_loss",loss.item(),"train_acc",acc.item())
            start = time.time()
        train_loss /= len(train_loader)
        train_acc1 /= len(train_loader)
        time_avg /= len(train_loader)
        data_avg /= len(train_loader)
        backward_avg /= len(train_loader)
        lr_scheduler.step()
        
        layer1.eval()
        layer2.eval()
        layer3.eval()
        if rank == 0:
            print("lr:",get_lr(optimizer))
        val_loss = 0.0
        val_acc1 = 0.0
        with torch.no_grad():
            for i, (image,label) in enumerate(val_loader):
                image = image.to(rank,non_blocking = True)
                label = label.to(rank,non_blocking = True)
                
                outputs =layer1(image)
                outputs = layer2(outputs)

                outputs = layer3(outputs)
                loss = criterion(outputs,label)
                acc,_ = accuracy(outputs,label,topk =(1,2))
    
                val_loss += loss.item()
                val_acc1 += acc.item()
                if i %20 == 0 and rank == 0:
                    print("val_loss",loss.item(),"val_acc",acc.item())
            val_loss /= len(val_loader)
            val_acc1 /= len(val_loader)
            print(len(val_loader))
        if rank == 0:
            print("epoch:",epoch,"train_loss",train_loss,"train_acc",train_acc1,"val_loss",val_loss,"val_acc",val_acc1)
            file_save = open(args.log_dir, mode="a")
            file_save.write(
                "\n"
                + "step:"
                + str(epoch)
                + "  loss_train:"
                + str(train_loss)
                + "  acc1_train:"
                + str(train_acc1)
                + "  loss_val:"
                + str(val_loss)
                + "  acc1_val:"
                + str(val_acc1)
                + "  time_per_batch:"
                + str(time_avg)
                + "  data_time:"
                + str(data_avg)
                + "  backward_time:"
                + str(backward_avg)
                + "  lr:"
                + str(get_lr(optimizer))
            )
            file_save.close()

            
if __name__ == '__main__':
    main()

