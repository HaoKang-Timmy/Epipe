import torch
import torch.nn as nn
from typing import List
from distributedlayers.defered_bn import DeferredBatchNorm
import torch.multiprocessing as mp

class dist_gpipe:
    def __init__(
        self, model_partition: List[nn.Sequential], devices: List[int],chunks:int,backend,init_method
    ) -> None:

        # for i,model in enumerate(model_partition):
        #     DeferredBatchNorm.convert_deferred_batch_norm(model, chunks)
        #     model = model.to(devices[i])
        num_devices = set(devices)# device number
        devices_index = []
        for i in num_devices:
            devices_index.append([j for j,x in enumerate(num_devices) if x == i])
        pass
        model_perdevice = []
        for i in num_devices:
            models = []
            for j in devices_index[i]:
                model_partition[j] = model_partition[j].to(num_devices[i])
                models.append(model_partition[j])
            # models = models.to(num_devices[i])
            model_perdevice.append(models)
        self.model_list = model_perdevice
        self.num_devices = len(num_devices)

    def run(input):
        mp.spawn()



        
            
