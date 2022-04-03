'''
Author: your name
Date: 2022-04-03 12:55:07
LastEditTime: 2022-04-03 14:49:24
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe_gloo/utils.py
'''
import time
import torch
import torch.distributed as dist
from compression import TopkLayer,QSendLayer,QRecvLayer
from torch.optim import AdamW, SGD
from transformers import get_scheduler
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
def init_models_client(train_settings,client_settings):
    param_list = []
    for model in train_settings['models']:
        model = model.to(train_settings['device'],non_blocking = True)
        param_list.append({"params":model.parameters()})
    if train_settings['tasktype'] == 'cv':
        optimizer = SGD(param_list,lr = train_settings['lr'],weight_decay=train_settings['wd'])
        warmup_scheduler = get_scheduler(name = 'cosine',optimizer = optimizer,num_warmup_steps = int(train_settings['epochs']/10),num_training_steps = train_settings['epochs'])
    elif train_settings['tasktype'] == 'nlp':
        optimizer = AdamW(param_list,lr = train_settings['lr'],weight_decay=train_settings['wd'])
        warmup_scheduler = get_scheduler(name = 'linear',optimizer = optimizer,num_warmup_steps = int(train_settings['epochs']/10),num_training_steps = train_settings['epochs'])
    topk_layer= TopkLayer(train_settings['prune'])
    quant_layer = QSendLayer(train_settings['quant'],client_settings['next_rank'],client_settings['rank'])
    dequant_layer = QRecvLayer(train_settings['quant'],client_settings['last_rank'],client_settings['rank'])
    return topk_layer, quant_layer, dequant_layer  
def trainer(train_settings,client_settings):
    acc1_avg = 0.0
    losses_avg = 0.0
    train_acc1_avg = 0.0
    train_losses_avg = 0.0
    time_per_batch = 0.0
    batch_time = 0.0
    start = time.time()
    if train_settings['tasktype'] == 'cv':
        for batch_iter, (images, targets) in enumerate(train_settings['train_loader']):
        

def client(train_settings,client_settings):
    torch.multiprocessing.set_sharing_strategy("file_system")
    print("process begin: ", client_settings['rank'])
    dist.init_process_group(backend = client_settings['backend'], init_method = client_settings['dist_url'],world_size= client_settings['world_size'],rank = client_settings['rank'])
    topk_layer, quant_layer, dequant_layer = init_models_client(train_settings,client_settings)
    for epoch in range(train_settings['epochs']):
        time,acc,metric = trainer()
