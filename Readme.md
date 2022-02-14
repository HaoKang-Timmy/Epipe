---
typora-copy-images-to: ./pic
---

# Gpipe with compression tests

# Introductions and settings

Gpipe is a technique that built for model parallel. It separates batches to micro-batches and transfer them while computing.

Here is a test about some compression tests.

## Settings

| Model             | MobileNetV2                                  |
| ----------------- | -------------------------------------------- |
| Dataset           | CIFAR10                                      |
| Training_strategy | train from scratch                           |
| lr_init           | 0.4                                          |
| Batch_size        | 1024                                         |
| Chunk             | 4(every batch is splited to 4 micro batches) |
| Optimizer         | SGD                                          |
| Momentum          | 0.9                                          |
| Weight_decay      | 1e-4                                         |
| Epochs            | 200                                          |
| Scheduler         | cosannealing with linear warp up(20 epochs)  |
| Pruning methods   | topk pruning, downsample,quantization        |

## Results



pruning rate(input_pruning/input_before_pruning) vs validation_acc1

![image-20220215015840995](/Users/catbeta/Documents/research/gpipe_test/pic/image-20220215015840995.png)

# code

```python
class TopkFunction(autograd.Function):
    @staticmethod
    def forward(ctx,input,ratio):
        shape = input.shape
        input =input.view(-1)
        src, index = torch.topk(torch.abs(input),int(ratio*input.shape[0]))
        mask = torch.zeros(input.shape).to(input.get_device())
        mask.index_fill_(0,index,1.0)
        input =input * mask
        mask =mask.view(shape)
        ctx.mask =mask
        input = input.view(shape)
        return input
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output * ctx.mask, None
class TopkAbs(nn.Module):
    def __init__(self,compress_ratio):
        super(TopkAbs,self).__init__()
        self.ratio = compress_ratio
    def forward(self,x):
        return TopkFunction.apply(x,self.ratio)



class Quantfunction(autograd.Function):
    @staticmethod
    def forward(ctx,input,floatpoint):
        ctx.floatpoint = floatpoint
        return input - input%floatpoint
    def backward(ctx,grad_output):
        return grad_output - grad_output%ctx.floatpoint,None
class Quant(nn.Module):
    def __init__(self,remain_float):
        super(Quant,self).__init__()  
        self.remain_float = remain_float
    def forward(self,x):
        return Quantfunction.apply(x,self.remain_float)
```

# different layers

