import time
import torch
import torch.nn as nn
from dataset.dataset_collection import DatasetCollection
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed as dist
from distributed_layers import Reshape1,Topk,TopkAbs,QuantLayer,DeQuantLayer
from distributed_layers import ForwardSend_BackwardReceive,ForwardReceive_BackwardSend,generate_recv
from torchgpipe.batchnorm import DeferredBatchNorm
from typing import cast,List
def prepare_dataloader(normalize, compose_train, compose_val, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset_collection = DatasetCollection(
        args.dataset_type, args.data, compose_train, compose_val)
    train_dataset, val_dataset = dataset_collection.init()


    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_sampler, train_loader, val_loader


def train_header(rank,model1,model2,optimizer,train_loader,criterion,args):
    # print("Use GPU",rank, "as the first part")
    model1.train()
    model2.train()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    acc1_avg = 0.0
    loss_avg = 0.0
    start = time.time()
    for k, (images, targets) in enumerate(train_loader):

        targets = targets.cuda(rank, non_blocking=True)
        
        images = images.cuda(rank, non_blocking=True)
        data_time = time.time() -start
        # images = images.chunk(args.chunks)
        # output = model(images)
        # output = ForwardSend_BackwardReceive.apply(output,rank+1,rank+1,rank)
        images = images.chunk(args.chunks)
        targets = targets.chunk(args.chunks)
        batches = list()
        for i in range(args.chunks):
            
            output = model1(images[i])
            output = ForwardSend_BackwardReceive.apply(output,rank+1,rank+1,rank)
            batches.append(output)
            # print("forward,part:",0,"device:",0,"chunk:",i)

        loss = list()
        acc1 = 0.
        for i in range(args.chunks):
            
            label = generate_recv(args.world_size-1,rank)
            label = ForwardReceive_BackwardSend.apply(label,args.world_size-1,args.world_size-1,rank)
            # print(label.shape)
            # label = model2(label)
            output = criterion(label,targets[i])
            loss.append(output)
            acc, _ = accuracy(label, targets[i], topk=(1, 5))
            acc1 = acc + acc1
            # print("forward,part:",4,"device:",rank,"chunk:",i)


        for i in range(args.chunks):
            loss[i].backward()
            # print("backward,part:",4,"device:",rank,"chunk:",i)
            optimizer.zero_grad()
            if i == 0:
                recv_size = batches[i].clone().detach()
            batches[i].backward(recv_size)
            # print("backward,part:",0,"device:",rank,"chunk:",i)
        optimizer.step()
        batch_time = time.time() - start
        # acc1, acc5 = accuracy(label, targets, topk=(1, 5))
        batch_time_avg = batch_time_avg + batch_time
        data_time_avg = data_time_avg + data_time
        loss_sum = 0.
        for j in range(len(loss)):
            loss_sum = loss_sum + loss[j].item()
        loss_avg = loss_avg + loss_sum
        if k % 30 == 0:
            print("train_loss:",loss_sum.item()/args.chunks,"train_acc1",acc1.item()/args.chunks)
        acc1_avg = acc1 + acc1_avg
        start = time.time()
    batch_time_avg = batch_time_avg / len(train_loader)
    data_time_avg =data_time_avg / len(train_loader)
    acc1_avg = acc1_avg / len(train_loader)
    loss_avg =loss_avg / len(train_loader)
    #TODO loss
    return batch_time_avg,data_time_avg,acc1_avg,loss_avg


def val_header(rank,model1,model2, val_loader,criterion,args):
    model1.eval()
    model2.eval()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    acc1_avg = 0.0
    loss_avg = 0.0
    #for i 
    with torch.no_grad():
        end = time.time()
        for _,(images,target) in enumerate(val_loader):
            images = images.cuda(rank, non_blocking = True)
            target = target.cuda(rank, non_blocking=True)
            images = images.chunk(args.chunks)
            # targets = targets.chunk(args.chunks)
            for i in range(args.chunks):
                output = model1(images[i])
                output = ForwardSend_BackwardReceive.apply(output,rank+1,rank+1,rank)
            label_scatter = torch.tensor([]).to(rank)
            loss_scatter = torch.tensor(0.)
            for i in range(args.chunks):
                label = generate_recv(args.world_size-1,rank)
                label = ForwardReceive_BackwardSend.apply(label,args.world_size-1,args.world_size-1,rank)
                # loss = criterion(label,target)
                # label = model2(label)
                label_scatter = torch.cat((label_scatter,label),0)
            loss = criterion(label_scatter,target)
            acc1, acc5 = accuracy(label_scatter, target, topk=(1, 5))
            batch_time = time.time() - end
            end = time.time()
            batch_time_avg = batch_time_avg + batch_time
            loss_avg = loss_avg + loss
            acc1_avg = acc1_avg + acc1
            if _%30 == 0:
                print("train_loss:",loss,"train_acc1",acc1)
    batch_time_avg = batch_time_avg / len(val_loader)
    acc1_avg = acc1_avg / len(val_loader)
    loss_avg =loss_avg / len(val_loader)
    return batch_time_avg,acc1_avg,loss_avg




    
def train_medium(rank,model,optimizer,iter_time,args):
    # print("Use GPU",rank, "as the medium part")
    model.train()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    start = time.time()
    for _ in range(iter_time):
        # print("rank:",rank,"i:",i,"begin")
        data_time = time.time() -start
        batches = list()
        for i in range(args.chunks):
            
            input = generate_recv(rank-1,rank)
            # print("forward,part:",rank,"device:",rank,"chunk:",i,"recv")
            input = ForwardReceive_BackwardSend.apply(input,rank-1,rank-1,rank)

            output = model(input)
            output = ForwardSend_BackwardReceive.apply(output,rank+1,rank+1,rank)
            batches.append(output)
            # print("forward,part:",rank,"device:",rank,"chunk:",i,"send")

        for i in range(args.chunks):

            optimizer.zero_grad()
            recv_size = batches[i].clone().detach()
            batches[i].backward(recv_size)
            # print("backward,part:",rank,"device:",rank,"chunk:",i)
        optimizer.step()
        batch_time = time.time() - start
        
        batch_time_avg = batch_time_avg + batch_time
        data_time_avg = data_time_avg + data_time
        start = time.time()
    batch_time_avg = batch_time_avg / iter_time
    data_time_avg =data_time_avg / iter_time
    return batch_time_avg,data_time_avg

def val_medium(rank,model,iter_time,args):
    batch_time_avg = 0.0
    model.eval()
    with torch.no_grad():
        end = time.time()
        for _ in range(iter_time):
            for i in range(args.chunks):
                input = generate_recv(rank-1,rank)
                input = ForwardReceive_BackwardSend.apply(input,rank-1,rank-1,rank)
                output = model(input)
                output = ForwardSend_BackwardReceive.apply(output,rank+1,rank+1,rank)
            batch_time = time.time() - end
            end = time.time()
            batch_time_avg = batch_time_avg + batch_time
        batch_time_avg = batch_time_avg / iter_time
        return batch_time_avg





def train_last(rank,model,optimizer,iter_time,args):
    # print("Use GPU",rank, "as the last part")
    # torch.autograd.set_detect_anomaly(True)
    torch.autograd.set_detect_anomaly(True)
    model.train()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    start = time.time()
    for _ in range(iter_time):
        # print("rank:",rank,"i:",i,"begin")
        # target = generate_recv(0,rank)
        # target = ForwardReceive_BackwardSend.apply(target,0,0,rank)
        # target = target.type(torch.int64)
        data_time = time.time() -start
        batches = list()
        for i in range(args.chunks):
            # print(rank)
            input = generate_recv(rank-1,rank)
            # print(rank)
            input = ForwardReceive_BackwardSend.apply(input,rank-1,rank-1,rank)
            # print(rank)
            output = model(input)
            
            output = ForwardSend_BackwardReceive.apply(output,0,0,rank)
            # print(rank)
            batches.append(output)
            # print("forward,part:",rank,"device:",rank,"chunk:",i)
        # loss = criterion(output,target)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        for i in range(args.chunks):
            output = batches[i]
            recv = output.clone().detach()
            optimizer.zero_grad()
            # print(output.shape)
            # print(recv.shape)
            # print(batches[i])
            output.backward(recv)
            # print("backward,part:",rank,"device:",rank,"chunk:",i)
        optimizer.step()
        batch_time = time.time() - start
        batch_time_avg = batch_time_avg + batch_time
        data_time_avg = data_time_avg + data_time
        # acc1_avg = acc1 + acc1_avg
        # if i% 30 == 0:
        #     print("loss:",loss,"acc1",acc1)
        start = time.time()
    batch_time_avg = batch_time_avg / iter_time
    data_time_avg =data_time_avg / iter_time
    return batch_time_avg,data_time_avg

def val_last(rank,model,iter_time,args):
    model.eval()
    batch_time_avg = 0.0
    with torch.no_grad():
        end = time.time()
        for _ in range(iter_time):
            for i in range(args.chunks):
                input = generate_recv(rank-1,rank)
                input = ForwardReceive_BackwardSend.apply(input,rank-1,rank-1,rank)
                output = model(input)
                output = ForwardSend_BackwardReceive.apply(output,0,0,rank)
            batch_time = time.time() -end
            end = time.time()
            batch_time_avg = batch_time_avg + batch_time
        batch_time_avg = batch_time_avg / iter_time
        return batch_time_avg




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
def train(model,train_loader,optimizer,criterion,args):
    model.train()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    acc1_avg = 0.0
    loss_avg = 0.0
    start = time.time()
    for i, (images, targets) in enumerate(train_loader):
        model.train()
        images = images.cuda(0, non_blocking = True)
        targets = targets.cuda(0, non_blocking=True)
        output = model(images)
        # print(output.shape,targets.shape)
        loss = criterion(output,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start
        
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_time_avg = batch_time_avg + batch_time
        loss_avg = loss_avg + loss
        if i % 30 == 0:
            print("train_loss:",loss,"train_acc1",acc1)
        acc1_avg = acc1 + acc1_avg
        start = time.time()
    batch_time_avg = batch_time_avg / len(train_loader)
    acc1_avg = acc1_avg / len(train_loader)
    loss_avg =loss_avg / len(train_loader)
    print("real",acc1_avg,loss_avg)
    return batch_time_avg,acc1_avg,loss_avg
def validate(model,val_loader,criterion,args):
    model.eval()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    acc1_avg = 0.0
    loss_avg = 0.0
    with torch.no_grad():
        end = time.time()
        for i,(images,target) in enumerate(val_loader):
            images = images.cuda(0, non_blocking = True)
            target = target.cuda(0, non_blocking=True)
            output = model(images)
            loss = criterion(output,target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_time = time.time() - end
            end = time.time()
            batch_time_avg = batch_time_avg + batch_time
            loss_avg = loss_avg + loss
            acc1_avg = acc1_avg + acc1
            if i%30 == 0:
                print("train_loss:",loss,"train_acc1",acc1)
        batch_time_avg = batch_time_avg / len(val_loader)
        acc1_avg = acc1_avg / len(val_loader)
        loss_avg =loss_avg / len(val_loader)
        return batch_time_avg,acc1_avg,loss_avg
def train_gpipe_header(rank,model,optimizer,train_loader,criterion,args):
    # print("Use GPU",rank, "as the first part")
    model.train()
    batch_time_avg = 0.0
    data_time_avg = 0.0
    acc1_avg = 0.0
    loss_avg = 0.0
    start = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # print("rank:",rank,"i:",i,"begin")
        # for j in range(args.world_size):
        #     dist.send(torch.tensor(targets[0]))
        targets = targets.cuda(rank, non_blocking=True)
        # targets = ForwardSend_BackwardReceive.apply(targets,args.world_size-1,args.world_size-1,rank)
        data_time = time.time() -start
        images = images.cuda(rank, non_blocking=True)
        output = model(images)
        output = ForwardSend_BackwardReceive.apply(output,rank+1,rank+1,rank)
        # optimizer.zero_grad()
        # recv_size = output.clone().detach()
        # output.backward(recv_size)
        # optimizer.step()
        label = generate_recv(args.world_size-1,rank)
        label = ForwardReceive_BackwardSend.apply(label,args.world_size-1,args.world_size-1,rank)
        loss = criterion(label,targets)
        loss.backward()
        optimizer.zero_grad()
        recv_size = output.clone().detach()
        output.backward(recv_size)
        optimizer.step()
        batch_time = time.time() - start
        acc1, acc5 = accuracy(label, targets, topk=(1, 5))
        batch_time_avg = batch_time_avg + batch_time
        data_time_avg = data_time_avg + data_time
        loss_avg = loss_avg + loss
        if i % 30 == 0:
            print("train_loss:",loss,"train_acc1",acc1)
        acc1_avg = acc1 + acc1_avg
        start = time.time()
    batch_time_avg = batch_time_avg / len(train_loader)
    data_time_avg =data_time_avg / len(train_loader)
    acc1_avg = acc1_avg / len(train_loader)
    loss_avg =loss_avg / len(train_loader)
    #TODO loss
    return batch_time_avg,data_time_avg,acc1_avg,loss_avg

def make_partition(model,num,type:str,pruning = None,scale_bits = None):
    if num == 4:
        print(type)
        if type.find("cloud") != -1:
            layer1 = [model.conv1,model.bn1]
            layer2 = [model.layers[0:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:],model.conv2]
            layer5 = [model.bn2,Reshape1(), model.linear]
            print("cloud")
        # if type.find("edge") != -1 :
        #     layer1 = [model.conv1,model.bn1,model.layers[0]]
        #     layer2 = [model.layers[1:7]]
        #     layer3 = [model.layers[7:14]]
        #     layer4 = [model.layers[14:]]
        #     layer5 = [model.conv2, model.bn2,Reshape1(), model.linear]
        #     print("edge")
        # if type.find("lastoned")!= -1:
        #     layer1 = [model.conv1,model.bn1]
        #     layer2 = [model.layers[0:7]]
        #     layer3 = [model.layers[7:14]]
        #     layer4 = [model.layers[14:]]
        #     layer5 = [model.conv2, model.bn2,Reshape1(), model.linear]
            # print("lastoned")
        if type.find("first2") != -1:
            layer1 = [model.conv1,model.bn1,model.layers[0]]
            layer2 = [model.layers[1:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:],model.conv2,]
            layer5 = [ model.bn2,Reshape1(), model.linear]
            print("first2")
        if type.find("first3") != -1:
            layer1 = [model.conv1,model.bn1,model.layers[0:2]]
            layer2 = [model.layers[2:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:],model.conv2,]
            layer5 = [ model.bn2,Reshape1(), model.linear]
            print("first3")
        if type.find("last2") != -1:
            layer1 = [model.conv1,model.bn1]
            layer2 = [model.layers[0:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:]]
            layer5 = [ model.conv2,model.bn2,Reshape1(), model.linear]
            print("last2")
        if type.find("last3") != -1:
            layer1 = [model.conv1,model.bn1]
            layer2 = [model.layers[0:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:-1]]
            layer5 = [ model.layers[-1],model.conv2,model.bn2,Reshape1(), model.linear]
            print("last3")
        if type.find("f2l2") !=-1:
            layer1 = [model.conv1,model.bn1,model.layers[0]]
            layer2 = [model.layers[1:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:]]
            layer5 = [ model.conv2,model.bn2,Reshape1(), model.linear]
        if type.find("f2l3") != -1:
            layer1 = [model.conv1,model.bn1,model.layers[0]]
            layer2 = [model.layers[0:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:-1]]
            layer5 = [ model.layers[-1],model.conv2,model.bn2,Reshape1(), model.linear]
        if type.find("f3l2") != -1:
            layer1 = [model.conv1,model.bn1,model.layers[0:2]]
            layer2 = [model.layers[2:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:]]
            layer5 = [ model.conv2,model.bn2,Reshape1(), model.linear]
        if type.find("f3l3") != -1:
            layer1 = [model.conv1,model.bn1,model.layers[0:2]]
            layer2 = [model.layers[2:7]]
            layer3 = [model.layers[7:14]]
            layer4 = [model.layers[14:-1]]
            layer5 = [ model.layers[-1],model.conv2,model.bn2,Reshape1(), model.linear]
        if type.find("pruning")!= -1:
            layer1.append(TopkAbs(pruning))
            layer4.append(TopkAbs(pruning))
            print("pruning")
        if type.find("quantization") != -1:
            layer1.append(QuantLayer(scale_bits))
            layer2.insert(0,DeQuantLayer(scale_bits))
            layer4.append(QuantLayer(scale_bits))
            layer5.insert(0,DeQuantLayer(scale_bits))
            print("quantization")
        layer1 = nn.Sequential(*layer1).to(0)
        layer2 = nn.Sequential(*layer2).to(1)
        layer3 = nn.Sequential(*layer3).to(2)
        layer4 = nn.Sequential(*layer4).to(3)
        layer5 = nn.Sequential(*layer5).to(0)
        partition = [layer1,layer2,layer3,layer4,layer5]
        partition = cast(List[nn.Sequential], nn.ModuleList(partition))
    return partition