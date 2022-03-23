
# from my_pipe import RemoteQuantizationLayer,RemoteDeQuantizationLayer
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import os
def main_worker(rank,nothing):
    # os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    print("begin",rank)
    pass
    dist.init_process_group(backend='gloo', init_method='tcp://18.25.6.30:10002',
                            world_size=2, rank=rank)
    print(rank)

main_worker(1,0)
