"""
Author: your name
Date: 2022-03-25 01:01:31
LastEditTime: 2022-03-25 01:04:57
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/test_type.py
"""
import torch.multiprocessing as mp
import torch
import torchvision.models as models
import torch.distributed as dist
from dist_gpipe import RemoteQuantizationLayer, RemoteDeQuantizationLayer


def testfunc(rank, nothing):
    print(rank)
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:1294",
        world_size=2,
        rank=rank,
        group_name="test",
    )
    if rank == 0:
        something = torch.rand([1, 2])
        something = something.type(torch.ShortTensor)

        dist.send(something, 1)
        print(something)
    if rank == 1:
        something = torch.rand([1, 2]).type(torch.ShortTensor)
        # something = torch.rand([1,2])
        dist.recv(something, 0)


def main():
    torch.multiprocessing.set_start_method("spawn")
    for i in range(2):
        # print(i)
        p = mp.Process(target=testfunc, args=(i, 1))
        p.start()


if __name__ == "__main__":
    main()
