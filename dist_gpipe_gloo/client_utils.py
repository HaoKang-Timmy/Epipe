'''
Author: your name
Date: 2022-04-03 12:55:07
LastEditTime: 2022-04-12 16:27:49
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe_gloo/utils.py
'''
#prune quant 这些不用的时候为0不是None
import time
import torch
import torch.distributed as dist
from .compression import TopkLayer,QSendLayerGPU,QRecvLayerGPU
from torch.optim import AdamW, SGD
from transformers import get_scheduler
import torch.nn as nn
from .utils import SendTensor,get_lr,accuracy,RecvTensor



def init_models_client(train_settings,client_settings):
    param_list = []
    for model in train_settings['models']:
        model = model.to(client_settings['device'],non_blocking = True)
        param_list.append({"params":model.parameters()})
    if train_settings['tasktype'] == 'cv':
        optimizer = SGD(param_list,lr = train_settings['lr'],weight_decay=train_settings['wd'])
        warmup_scheduler = get_scheduler(name = 'cosine',optimizer = optimizer,num_warmup_steps = int(train_settings['epochs']/10),num_training_steps = train_settings['epochs'])
    elif train_settings['tasktype'] == 'nlp':
        optimizer = AdamW(param_list,lr = train_settings['lr'],weight_decay=train_settings['wd'])
        warmup_scheduler = get_scheduler(name = 'linear',optimizer = optimizer,num_warmup_steps = int(train_settings['epochs']/10*train_settings['trainloader']),num_training_steps = train_settings['epochs']*train_settings['trainloader'])
    topk_layer = TopkLayer(train_settings['prune'],client_settings['send_size'])
    quant_layer = QSendLayerGPU(train_settings['quant'],client_settings['send_rank'],client_settings['rank'])
    dequant_layer = QRecvLayerGPU(train_settings['quant'],client_settings['recv_rank'],client_settings['rank'])
    criterion = nn.CrossEntropyLoss().to(client_settings['device'])
    return topk_layer, quant_layer, dequant_layer,optimizer ,warmup_scheduler, criterion


def client_trainer(train_settings,client_settings, topk_layer, quant_layer, dequant_layer,optimizer ,warmup_scheduler,criterion):
    acc1_avg = 0.0
    losses_avg = 0.0
    train_acc1_avg = 0.0
    train_losses_avg = 0.0
    time_per_batch = 0.0
    batch_time = 0.0
    if train_settings['tasktype'] == 'cv':
        for batch_iter, (images, targets) in enumerate(train_settings['train_loader']):
            start = time.time()
            images = images.to(client_settings['device'], non_blocking=True)
            targets = targets.to(client_settings['device'], non_blocking=True)
            images = images.chunk(client_settings['chunks'])
            targets = targets.chunk(client_settings['chunks'])
            batches = []
            acc1 = 0.0
            losses = 0.0
            for i, model in enumerate(train_settings['models']):
                batch = []
                model.train()
                for chunk in range(client_settings['chunks']):
                    if i == 0:
                        output = model(images[chunk])
                        # print("client",client_settings['rank'],"pre_send",output.shape)
                        if train_settings['prune'] !=0:
                            output = SendTensor(output,client_settings['send_rank'],client_settings['rank'],prune_layer = topk_layer)
                        if train_settings['sortquant'] != 0:
                            output = SendTensor(output,client_settings['send_rank'],client_settings['rank'],sortquant= 1,bits=train_settings['quant'],split = train_settings['split'])
                        elif train_settings['quant'] != 0:
                            # print(client_settings['rank'],"send quant")
                            output = SendTensor(output,client_settings['send_rank'],client_settings['rank'],quant_layer= quant_layer)
                        else:
                            output = SendTensor(output,client_settings['send_rank'],client_settings['rank'],send_layer=1)
                        
                        # print("client",client_settings['rank'],"send",output.shape)
                    else:
                        input = torch.zeros(client_settings["recv_size"]).to(client_settings['device']).requires_grad_()
                        # print("client",client_settings['rank'],"pre_recv",client_settings['recv_rank'],input.shape)
                        if train_settings['sortquant'] != 0:
                            input = RecvTensor(input,client_settings['recv_rank'],client_settings['rank'],sortdequant= 1,bits=train_settings['quant'],split = train_settings['split'])
                        elif train_settings['quant'] != 0:
                            input = RecvTensor(input,client_settings['recv_rank'],client_settings['rank'],dequant_layer= dequant_layer)
                        else:
                            input = RecvTensor(input,client_settings['recv_rank'],client_settings['rank'],recv_layer= 1)
                        # print("client",client_settings['rank'],"recv",client_settings['recv_rank'],input.shape)
                        output = model(input)
                        acc,_ = accuracy(output, targets[chunk],topk=(1,2))
                        output = criterion(output,targets[chunk])
                        losses = output.item()
                        acc1 = acc1 + acc.item()
                    batch.append(output)
                batches.append(batch)
            acc1 = acc1 / client_settings['chunks']
            losses = losses / client_settings['chunks']
            acc1_avg, losses_avg = acc1 + acc1_avg, losses_avg + losses
            if batch_iter % client_settings["showperiod"] == 0:
                print("tarining_loss:", losses, "training_acc", acc)
            

            for back in range(len(train_settings['models']) - 1, -1, -1):
                if back == len(train_settings['models']) - 1:
                    for chunk in range(client_settings['chunks']):
                        batches[back][chunk].backward()
                else:
                    for chunk in range(client_settings['chunks']):
                        batches[back][chunk].backward(
                            torch.empty(tuple(list(batches[back][chunk].shape))).to(client_settings['device'])
                        )
            optimizer.step()
            optimizer.zero_grad()
            batch_time = time.time() - start
            time_per_batch += batch_time
        time_per_batch = time_per_batch / len(train_settings['train_loader'])
        warmup_scheduler.step()
        train_acc1_avg, train_losses_avg = (
            acc1_avg / len(train_settings['train_loader']),
            losses_avg / len(train_settings['train_loader']),
        )
 
        return time_per_batch, train_acc1_avg,train_acc1_avg,train_losses_avg
                            
def client_validation(train_settings,client_settings,topk_layer, quant_layer, dequant_layer,criterion):
    if train_settings['tasktype'] == 'cv':
        for model in train_settings['models']:
            model.eval()
        val_acc_avg = 0.0
        val_loss_avg = 0.0
        with torch.no_grad():
            for batch_iter, (images, targets) in enumerate(train_settings['valloader']):
                images = images.to(client_settings['device'], non_blocking=True)
                targets = targets.to(client_settings['device'], non_blocking=True)
                images = images.chunk(client_settings['chunks'])
                targets = targets.chunk(client_settings['chunks'])
                acc1 = 0.0
                losses = 0.0
                for i, model in enumerate(train_settings['models']):
                    for chunk in range(client_settings['chunks']):
                        if i == 0:
                            output = model(images[chunk])
                            if train_settings['prune'] !=0:
                                output = SendTensor(output,client_settings['send_rank'],client_settings['rank'],prune_layer = topk_layer)
                            if train_settings['sortquant'] != 0:
                                output = SendTensor(output,client_settings['send_rank'],client_settings['rank'],sortquant= 1,bits=train_settings['quant'],split = train_settings['split'])
                            elif train_settings['quant'] != 0:
                                # print(client_settings['rank'],"send quant")
                                output = SendTensor(output,client_settings['send_rank'],client_settings['rank'],quant_layer= quant_layer)
                            else:
                                output = SendTensor(output,client_settings['send_rank'],client_settings['rank'],send_layer=1)
                        else:

                            input = (
                                torch.empty(client_settings["recv_size"]).to(client_settings['device']).requires_grad_()
                            )
                            if train_settings['sortquant'] != 0:
                                input = RecvTensor(input,client_settings['recv_rank'],client_settings['rank'],sortdequant= 1,bits=train_settings['quant'],split = train_settings['split'])
                            elif train_settings['quant'] != 0:
                                input = RecvTensor(input,client_settings['recv_rank'],client_settings['rank'],dequant_layer= dequant_layer)
                            else:
                                input = RecvTensor(input,client_settings['recv_rank'],client_settings['rank'],recv_layer= 1)
                            output = model(input)
                            acc, _ = accuracy(output, targets[chunk],topk = (1,2))
                            output = criterion(output, targets[chunk])
                            losses = losses + output.item()
                            acc1 = acc1 + acc.item()
                acc1 = acc1 / client_settings['chunks']
                losses = losses / client_settings['chunks']
                if batch_iter % client_settings["showperiod"] == 0:
                    print("val_loss:", losses, "val_acc:", acc1)
                val_acc_avg, val_loss_avg = acc1 + val_acc_avg, losses + val_loss_avg
            val_acc_avg, val_loss_avg = (
            val_acc_avg / len(train_settings['valloader']),
            val_loss_avg / len(train_settings['valloader']),
            )
            return val_acc_avg, val_acc_avg,val_loss_avg

def client(train_settings,client_settings):
    torch.cuda.set_device(client_settings["device"])
    #may be not right
    torch.multiprocessing.set_sharing_strategy("file_system")
    print("client",client_settings['backend'],client_settings['dist_url'],client_settings['world_size'],client_settings['rank'])
    dist.init_process_group(backend = client_settings['backend'], init_method = client_settings['dist_url'],world_size= client_settings['world_size'],rank = client_settings['rank'])
    print("process begin: ", client_settings['rank'])
    topk_layer, quant_layer, dequant_layer,optimizer ,warmup_scheduler, criterion = init_models_client(train_settings,client_settings)
    for epoch in range(train_settings['epochs']):
        train_time,train_acc,train_metric,train_loss = client_trainer(train_settings,client_settings,topk_layer, quant_layer, dequant_layer,optimizer ,warmup_scheduler,criterion)
        if train_settings['tasktype'] == 'cv':
            warmup_scheduler.step()
        val_acc,val_metric,val_loss = client_validation(train_settings,client_settings,topk_layer, quant_layer, dequant_layer,criterion)
        print(
            "epoch",
            epoch,
            "train_acc",
            train_acc,
            "train_loss",
            train_loss,
            "val_acc",
            val_acc,
            "val_loss",
            val_loss,
            "lr",
            get_lr(optimizer),
        )
        file_save = open(client_settings["savepath"], mode="a")
        file_save.write(
            "\n"
            + "step:"
            + str(epoch)
            + "  loss_train:"
            + str(train_loss)
            + "  acc1_train:"
            + str(train_acc)
            + "  loss_val:"
            + str(val_loss)
            + "  acc1_val:"
            + str(val_acc)
            + "  time_per_batch:"
            + str(train_time)
            + "  lr:"
            + str(get_lr(optimizer))
        )

