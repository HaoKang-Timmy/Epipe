from compression_layer_nccl import *
class CompressionClientSendLayer(nn.Module):
    def __init__(self,input,q,send_rank,device,bits,split_bits,pg = None) -> None:
        super(CompressionClientSendLayer, self).__init__()
        self.min_step = torch.