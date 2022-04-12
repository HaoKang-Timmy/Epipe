'''
Author: your name
Date: 2022-04-03 11:38:29
LastEditTime: 2022-04-12 16:28:02
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research/gpipe_test/dist_gpipe_gloo/dist_gpipe.py
'''
from .server_utils import server
from .client_utils import client
from .utils import make_dictions
import torch
import torch.multiprocessing as mp
class dist_gpipe:
    def __init__(self,args,model_list,devices,tensor_size) -> None:
        torch.multiprocessing.set_start_method("spawn")
        client_settings = {}
        server_settings_list = []
        client_train_settings = {}
        server_train_settings_list = []

        make_dictions(client_train_settings,client_settings,server_train_settings_list,server_settings_list,args,model_list,devices,tensor_size)
        self.client_settings = client_settings
        self.server_settings_list = server_settings_list
        self.client_train_settings = client_train_settings
        self.server_train_settings_list = server_train_settings_list
        self.num_devices = args.devices
        # # print(self.client_settings)
        # print(self.server_settings_list[1])
        # # print(self.client_train_settings)
        # print(self.server_train_settings_list[1])
    def session(self):
        processes = []
        for i in range(len(self.num_devices)):
            if i == 0:
                p = mp.Process(target = client, args = (self.client_train_settings,self.client_settings))
            else:
                p = mp.Process(target = server, args = (self.server_train_settings_list[i - 1],self.server_settings_list[i - 1]))
            p.start()
            processes.append(p)
        for process in processes:
            process.join()
