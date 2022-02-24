import torch
import torch.nn as nn
from typing import List


class dist_gpipe:
    def __init__(
        self, model_partition: List[nn.Sequential], devices: List[torch.device]
    ) -> None:
        pass
