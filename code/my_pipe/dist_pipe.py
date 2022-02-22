import torch
import torch.nn as nn
from typing import List
class DistGpipe:
    def __init__(
        self,
        model_list:List[nn.Sequential],
        devices:List[int],
        chunks:int = 1,
    )-> None:
        self.chunks = chunks
        self.devices = devices
        self.model_list = model_list
        self.num_device = len(set(devices))
        
    