'''
Author: your name
Date: 2022-04-08 18:57:41
LastEditTime: 2022-04-12 16:26:11
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe_gloo/compression/test.py
'''
from doctest import OutputChecker
import torch.multiprocessing as mp
from compression_test import QSendLayerGPU,QRecvLayerGPU,SortQuantGPU,SortDeQuantGPU
import torch.distributed as dist
import torch

def main():
    mp.spawn(main_worker, nprocs=2,
                 args=(4,4))
def relative_error(origin, quant):
    first = torch.abs(origin)
    second = torch.abs(quant)
    # print(torch.mean(torch.abs((torch.abs(origin) - torch.abs(quant)))/torch.abs(origin)))
    # print(((first - second)/first))
    # shape = first.view(-1).shape[0]
    # # print(((first - second)/first))
    # print(torch.isnan(first).any())
    # print(torch.isnan(second).any())
    # print(torch.isnan(first - second).any())
    # print(torch.isnan(first / second).any())
    # print(torch.isnan(torch.abs((first - second)/first)).any())
    return torch.abs(torch.abs((first - second))/first).mean()
def error(origin, quant):
    return torch.abs(torch.abs(origin) - torch.abs(quant)).sum()
def main_worker(rank,nothing,nothing1):
    
    torch.backends.cudnn.benckmark = False
    print(rank)
    if rank == 0:
        dist.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:1277',
                                    world_size=2, rank=rank)
        input = torch.rand([10,10]).to(0).requires_grad_()
        other = torch.rand([10,10]).to(0)
        # print(rank,input)
        output = SortQuantGPU.apply(input,6,2,1)
        # print(output.shape)
        output.backward(other)
        # dist.send(input,1)
        # output = layer(input)
        # dist.send(input,1)
        # dist.isend(input,1)
        # grad = torch.rand([4,4]).to(0)
        # output.backward(grad)
        # dist.send(input,1)
        # print(rank,input)
    elif rank == 1:
        dist.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:1277',
                                    world_size=2, rank=rank)
        # input = torch.rand([10,10]).to(1)
        # dist.recv(input,0)
        other = torch.rand([10,10]).to(1)
        input = torch.rand([10,10]).to(1).requires_grad_()
        output = SortDeQuantGPU.apply(input,6,2,0)
        output.backward(other)
        # print(rank,output)
        # dist.recv(other,0)
        # print(error(other,output))
        # # dist.recv(input,0)
        # output = layer(input)
        # dist.recv(other,0)
       # print(other)
        # print(rank,input)
        # some = relative_error(other,output)
        # print(error(other,output))
        # grad = torch.rand([4,4]).to(1)
        # output.backward(grad)
        # print(rank,grad)











if __name__ == '__main__':
    main()