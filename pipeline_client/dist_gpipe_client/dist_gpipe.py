from .client_utils import client
from .utils import make_dictions
import torch
import torch.multiprocessing as mp


class dist_gpipe_client:
    def __init__(self, args, model_list, tensor_size, train_loader, val_loader) -> None:

        client_settings, client_train_settings = make_dictions(
            args,
            model_list,
            tensor_size,
            train_loader,
            val_loader,
        )
        self.client_settings = client_settings
        self.client_train_settings = client_train_settings
        # print("client settings",self.client_settings)
        # print("server settings",self.server_settings_list)
        # print("client train settings",self.client_train_settings)
        # print("server train settings",self.server_train_settings_list)

    def session(self):
        processes = []

        p = mp.Process(
            target=client,
            args=(self.client_train_settings, self.client_settings),
        )

        p.start()
        processes.append(p)
        for process in processes:

            process.join()
            print("client finished")
