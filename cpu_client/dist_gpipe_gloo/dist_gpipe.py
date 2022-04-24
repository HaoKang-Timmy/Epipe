"""
Author: your name
Date: 2022-04-03 11:38:29
LastEditTime: 2022-04-12 19:18:55
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe_gloo/dist_gpipe.py
"""
from .server_utils import server
from .client_utils import client
from .utils import make_dictions
import torch
import torch.multiprocessing as mp


class dist_gpipe:
    def __init__(
        self, args, model_list, devices, tensor_size, train_loader, val_loader
    ) -> None:
        torch.multiprocessing.set_start_method("spawn")
        client_settings = {}
        server_settings_list = []
        client_train_settings = {}
        server_train_settings_list = []

        make_dictions(
            client_train_settings,
            client_settings,
            server_train_settings_list,
            server_settings_list,
            args,
            model_list,
            devices,
            tensor_size,
            train_loader,
            val_loader,
        )
        self.client_settings = client_settings
        self.server_settings_list = server_settings_list
        self.client_train_settings = client_train_settings
        self.server_train_settings_list = server_train_settings_list
        self.num_devices = args.devices
        # print("client settings",self.client_settings)
        # print("server settings",self.server_settings_list)
        # print("client train settings",self.client_train_settings)
        # print("server train settings",self.server_train_settings_list)

    def session(self):
        processes = []
        for i in range(len(self.num_devices)):
            if i == 0:
                p = mp.Process(
                    target=client,
                    args=(self.client_train_settings, self.client_settings),
                )
            else:
                p = mp.Process(
                    target=server,
                    args=(
                        self.server_train_settings_list[i - 1],
                        self.server_settings_list[i - 1],
                    ),
                )
            p.start()
            processes.append(p)
        for process in processes:

            process.join()
            print("ok")
