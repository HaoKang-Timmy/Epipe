'''
Author: your name
Date: 2022-04-09 01:26:19
LastEditTime: 2022-04-12 16:28:12
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe_gloo/server_utils.py
'''
import enum
import time
import torch
import torch.distributed as dist
from .compression import TopkLayer,QSendLayerGPU,QRecvLayerGPU
from torch.optim import AdamW, SGD
from transformers import get_scheduler
import torch.nn as nn
from .utils import SendTensor,get_lr,accuracy,RecvTensor
def init_models_server(train_settings,server_settings):
    param_list = []
    for model in train_settings['models']:
        model = model.to(train_settings['device'],non_blocking = True)
        # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[server_settings["device"]])
        param_list.append({"params":model.parameters()})
    if train_settings['tasktype'] == 'cv':
        optimizer = SGD(param_list,lr = train_settings['lr'],weight_decay=train_settings['wd'])
        warmup_scheduler = get_scheduler(name = 'cosine',optimizer = optimizer,num_warmup_steps = int(train_settings['epochs']/10),num_training_steps = train_settings['epochs'])
    elif train_settings['tasktype'] == 'nlp':
        optimizer = AdamW(param_list,lr = train_settings['lr'],weight_decay=train_settings['wd'])
        warmup_scheduler = get_scheduler(name = 'linear',optimizer = optimizer,num_warmup_steps = int(train_settings['epochs']/10*train_settings['trainloader']),num_training_steps = train_settings['epochs']*train_settings['trainloader'])
    topk_layer= TopkLayer(train_settings['prune'],server_settings['send_size'])
    quant_layer = QSendLayerGPU(train_settings['quant'],server_settings['send_rank'],server_settings['rank'])
    dequant_layer = QRecvLayerGPU(train_settings['quant'],server_settings['recv_rank'],server_settings['rank'])
    return topk_layer, quant_layer, dequant_layer,optimizer ,warmup_scheduler

def server_trainer(train_settings,server_settings,topk_layer, quant_layer, dequant_layer,optimizer ,warmup_scheduler):
    if train_settings['tasktype'] == 'cv':
        for batch_iter in range(train_settings["len_trainloader"]):
            batches = []
            for i, model in enumerate(train_settings['models']):
                model.train()
                batch = []
                for chunk in range(server_settings['chunks']):
                    input = torch.zeros(server_settings['recv_size']).to(server_settings["device"]).requires_grad_()
                    # print("server",server_settings['rank'],"pre_recv",server_settings['recv_rank'],input.shape)
                    if server_settings['recv_rank'] == 0:
                        if train_settings['sortquant'] != 0:
                            input = RecvTensor(input,server_settings['recv_rank'],server_settings['rank'],sortdequant= 1,bits=train_settings['quant'],split = train_settings['split'])
                        elif train_settings['quant'] != 0:
                            input = RecvTensor(input,server_settings['recv_rank'],server_settings['rank'],dequant_layer= dequant_layer)
                        else:
                            input = RecvTensor(input,server_settings['recv_rank'],server_settings['rank'],recv_layer= 1)          
                    else:
                        input = RecvTensor(input,server_settings['recv_rank'],server_settings['rank'],recv_layer= 1)
                    # print("server",server_settings['rank'],"recv",server_settings['recv_rank'],input.shape)
                    output = model(input)
                    # print("server",server_settings['rank'],"pre_send",server_settings['send_rank'],output.shape)
                    if server_settings['send_rank'] == 0:
                        if train_settings['prune'] != 0:
                            output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],prune_layer= topk_layer)
                        if train_settings['sortquant'] != 0:
                            output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],sortquant= 1,bits=train_settings['quant'],split = train_settings['split'])
                        elif train_settings['quant'] != 0:
                            output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],quant_layer= quant_layer)
                        else:
                            output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],send_layer=1)
                    else:
                        output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],send_layer= 1)
                    # print("server",server_settings['rank'],"send",server_settings['send_rank'],output.shape)
                    batch.append(output)
                batches.append(batch)
            for back in range(len(train_settings['models']) - 1, -1, -1):
                for chunk in range(server_settings['chunks']):

                    batches[back][chunk].backward(
                        torch.empty(tuple(list(batches[back][chunk].shape))).to(server_settings["device"])
                    )
            optimizer.step()
            optimizer.zero_grad()

def server_validation(train_settings,server_settings,topk_layer, quant_layer, dequant_layer):
    if train_settings['tasktype'] == 'cv':
        for model in train_settings['models']:
            model.eval()
        val_acc_avg = 0.0
        val_loss_avg = 0.0
        with torch.no_grad():
            for batch_iter in range(train_settings['len_valloader']):
              for i, model in enumerate(train_settings['models']):
                  model.eval() 
                  for chunk in range(server_settings['chunks']):
                        input = torch.zeros(server_settings['recv_size']).to(server_settings['device'])
                        if server_settings['recv_rank'] == 0:
                            if train_settings['sortquant'] != 0:
                                input = RecvTensor(input,server_settings['recv_rank'],server_settings['rank'],sortdequant= 1,bits=train_settings['quant'],split = train_settings['split'])
                            elif train_settings['quant'] != 0:
                                input = RecvTensor(input,server_settings['recv_rank'],server_settings['rank'],dequant_layer= dequant_layer)
                            else:
                                input = RecvTensor(input,server_settings['recv_rank'],server_settings['rank'],recv_layer= 1)          
                        else:
                            input = RecvTensor(input,server_settings['recv_rank'],server_settings['rank'],recv_layer= 1)
                        output = model(input)
                        if server_settings['send_rank'] == 0:
                            if train_settings['prune'] != 0:
                                output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],prune_layer= topk_layer)
                            if train_settings['sortquant'] != 0:
                                output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],sortquant= 1,bits=train_settings['quant'],split = train_settings['split'])
                            elif train_settings['quant'] != 0:
                                output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],quant_layer= quant_layer)
                            else:
                                output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],send_layer=1)
                        else:
                            output = SendTensor(output,server_settings['send_rank'],server_settings['rank'],send_layer= 1)



def server(train_settings,server_settings):
    torch.cuda.set_device(server_settings["device"])
    #maybe not right
    torch.multiprocessing.set_sharing_strategy("file_system")
    print("server",server_settings['backend'],server_settings['dist_url'],server_settings['world_size'],server_settings['rank'])
    dist.init_process_group(backend = server_settings['backend'], init_method = server_settings['dist_url'],world_size= server_settings['world_size'],rank = server_settings['rank'])
    print("process begin: ", server_settings['rank'])
    topk_layer, quant_layer, dequant_layer,optimizer ,warmup_scheduler= init_models_server(train_settings,server_settings)
    for epoch in range(train_settings['epochs']):
        server_trainer(train_settings,server_settings,topk_layer, quant_layer, dequant_layer,optimizer ,warmup_scheduler)
        if train_settings['tasktype'] == 'cv':
            warmup_scheduler.step()
        server_validation(train_settings,server_settings,topk_layer, quant_layer, dequant_layer)