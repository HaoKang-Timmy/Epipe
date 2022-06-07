"""
Author: your name
Date: 2022-04-03 11:38:29
LastEditTime: 2022-04-12 19:18:55
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe_gloo/dist_gpipe.py
"""
from .server_utils import server

from .utils import make_dictions_server
import torch
import torch.multiprocessing as mp


class dist_gpipe_server:
    def __init__(
        self, args, model, device, tensor_size, len_trainloader, len_valloader
    ) -> None:

        server_settings, server_train_settings = make_dictions_server(
            args, model, device, tensor_size, len_trainloader, len_valloader,
        )
        self.server_settings = server_settings
        self.server_train_settings = server_train_settings
        # self.num_devices = args.devices
        # print("client settings",self.client_settings)
        # print("server settings",self.server_settings_list)
        # print("client train settings",self.client_train_settings)
        # print("server train settings",self.server_train_settings_list)

    def session(self):
        processes = []

        p = mp.Process(
            target=server, args=(self.server_train_settings, self.server_settings),
        )
        p.start()
        processes.append(p)
        for process in processes:

            process.join()
            print("server finished")
