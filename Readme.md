---
typora-copy-images-to: ./pic
---

# TO DO LIST

## What I have done

### 1.distgpipe training

On CIFAR10 MobilenetV2 training

Traditional quantization is one step, one min per tensor. Multiple quantization has multiple steps and mins

Here are some important data. And I have done some train efficiency tests on dist-gpipe, which speed up the process at least 30%.

CIFAR10

| Training method         | Compression method                    | Acc%       |
| ----------------------- | ------------------------------------- | ---------- |
| tfs(train from scratch) | No                                    | 94.07%     |
| tfs(train from scratch) | Quantization16(traditional)           | 93.94%     |
| tfs(train from scratch) | Prune 0.5                             | 94.02%     |
| tfs(train from scratch) | Prune 0.1                             | 90.1%      |
| Finetune                | No                                    | 95.9%~0.1% |
| Finetune                | Quantization16(traditional)           | 95.8%~0.1% |
| Finetune                | Prune 0.5                             | 95.9%~0.1% |
| Finetune                | Prune 0.2                             | 95.1%~0.1% |
| Finetune                | Prune 0.1                             | 94.3%      |
| Finetune                | Sort Quant12(9bits quant 3bits split) | 95.7%      |
| Finetune                | Sort Quant8(6bits quant 2bits split)  | 95.7%      |
| Finetune                |                                       |            |

RTE

| Training method | Compression method             | Acc%       |
| --------------- | ------------------------------ | ---------- |
| Finetune        | No                             | 79.7%~0.2% |
| Finetune        | Quant6                         | 75.1%~3.2% |
| Finetune        | Sort Quant6(4bits 2bits split) | 77.4%~2.0% |
| Finetune        | Quant8                         | 78.4%~1%   |
| Finetune        | Sort Quant8(6bits 2bits split) | 78.4%~1%   |
|                 |                                |            |

### 2.Dataparallel tests

I have trained 100epochs here.

On CIFAR10 MobilenetV2 training

| Training method | Compression method                      | Acc%        |
| --------------- | --------------------------------------- | ----------- |
| Finetune        | No                                      | 96.1%       |
| Finetune        | Quantization 16bits                     | 96.0%       |
| Finetune        | Quantization 12bits                     | 95.4%       |
| Finetune        | Quantization 11bits                     | 91% 10%     |
| Finetune        | Quantization 10bits                     | 68%         |
| Finetune        | Quantization 9bits                      | 68%         |
| Finetune        | Prune0.5                                | 96.1%       |
| Finetune        | Prune 0.2                               | 95.2%       |
| Finetune        | Quantization 11bits                     | 91.3% 84.5% |
| Finetune        | SortQuantization 8bits 8splits          | 95.6%       |
| Finetune        | SortQuantization 8bits 8splits prune0.5 | 95.6%       |
| Finetune        | SortQuantization 4bits 16splits         | 95.3%       |
|                 |                                         |             |

The reason that quantization 11bits has two acc is that, it's curve first climb quickly like quantization 16bits but suddenly fall to 60% and then climb slowly.

On NLP tasks, for the cola dataset, I use Matthew's correlation. The rte dataset uses **validation acc**.

| Tasks | Training method | Compression method              | Validation_value |
| ----- | --------------- | ------------------------------- | ---------------- |
| Cola  | Sota            | None                            | 0.636            |
| Cola  | Finetune        | None                            | 0.634～0.008     |
| Cola  | Finetune        | Prune 0.5                       | 0.633～0.013     |
| Cola  | Finetune        | Quantization 16                 | 0.632～0.010     |
| Cola  | Finetune        | Quantization 10                 | 0.635~0.014      |
| Cola  | Finetune        | Quantization 8                  | 0.644~0.001      |
| Cola  | Finetune        | Quantization 4                  | 0(acc: 69.1%)    |
| RTE   | Sota            | None                            | 78.9%            |
| RTE   | Finetune        | None                            | 78.4% ~ 0.6%     |
| RTE   | Finetune        | Prune 0.5                       | 79.3%~ 0.7%      |
| RTE   | Finetune        | Quantization 16                 | 78.7% ~ 0.7%     |
| RTE   | Finetune        | Quantization 10                 | 0.783~1.1%       |
| RTE   | Finetune        | Quantization 8                  | 77.5% ~ 0.8%     |
| RTE   | Finetune        | Sort Quantization 6bits 4splits | 79.5% ~0.5%      |
| RTE   | Finetune        | Quantization 4                  | 52.2% ~0.1%      |

## To do

1. Finish multiple quantization in distributed method and run samples of CIFAR10 using MobileNetV2 backend.

   motivation: try this new method on the dist_gpipe system, test how many sets of steps and min are reasonable for **8bits** quantization

2. Fulfill the difference of input and output of quantization functions, record them and analyze.

   Motivation: analyze why the error will increase during small bits quantization training. And How to decrease it? Any specification that could represent the tensors distance that could give a response answer to acc decay dramatically.

3. Finish Roberta classification tasks using (prun ,quant,multiple quant) and analyze the results

   Motivation: tests compression layers on NLP tasks(don't prun nn.Embedding again).

# new method introduce

Haokang_quantization

Here is the pseudocode

```python
class SortQuantization(autograd.Function):
    @staticmethod
    def forward(ctx,input,bits,ratio,partition):
        shape = input.shape
        test = input
        input = input.view(-1)
        mask = torch.zeros(input.shape).to(input.get_device())
        src, index = torch.topk(torch.abs(input), int(ratio * input.shape[0]))
        # index and src to send 
        mask.index_fill_(0, index, 1.0)
        input = input * mask
        src = input.index_select(0,index)
        src1, index1 = torch.sort(src, dim = 0,descending=True)
        index1 = index1.chunk(partition)
        src1 = src1.chunk(partition)
        for i in range(partition):
            min, max = src1[i].min(),src1[i].max()
            if min != max:
                step = (max - min) / (pow(2, bits) - 1)
                temp_src = torch.round((src1[i] - min) / step) - pow(2, bits - 1)
                temp_src = (temp_src + pow(2, bits - 1)) * step + min
            else:
                temp_src = src1[i]
            src.scatter_(0,index1[i],temp_src)
        input.scatter_(0,index,src)
        ctx.mask = mask.view(shape)
        ctx.ratio = ratio
        ctx.bits = bits
        ctx.partition = partition
        input = input.view(shape)
        # if input.get_device() == 0:
        #     print("forward",torch.abs(torch.abs(input) - torch.abs(test)).sum()/torch.abs(test).sum())
        return input
    @staticmethod
    def backward(ctx,grad_backward):
        test = grad_backward
        shape = grad_backward.shape
        grad_backward = grad_backward * ctx.mask
        grad_backward = grad_backward.view(-1)
        index = grad_backward.nonzero()
        index = index.view(-1)
        src = grad_backward.index_select(0,index)
        src = src.view(-1)
        src1, index1 = torch.sort(src, dim = 0,descending=True)
        index1= index1.chunk(ctx.partition)
        src1 = src1.chunk(ctx.partition)
        for i in range(ctx.partition):
            min, max = src1[i].min(),src1[i].max()
            if min != max:
                step = (max - min) / (pow(2, ctx.bits) - 1)
                src_temp = torch.round((src1[i] - min) / step) - pow(2, ctx.bits - 1)
                src_temp = (src_temp + pow(2, ctx.bits - 1)) * step + min
            else:
                src_temp = src1[i]
            src.scatter_(0,index1[i],src_temp)
        grad_backward.scatter_(0,index,src)
        grad_backward = grad_backward.view(shape)
        return grad_backward,None,None,None
```

# comparing to k-means

| Settings                    | Method                                  | Input size                        | Time per batch | Acc                          |
| --------------------------- | --------------------------------------- | --------------------------------- | -------------- | ---------------------------- |
| CIFAR10 MobileNetV2 10epoch | K-means 4bits(20 iter)                  | [16,24,56,56]                     | 0.66s          | 93.01%                       |
| CIFAR10 MobileNetV2 10epoch | K-means 4bits(50 iter)                  | [16,24,56,56]                     | 1.33s          | 93.17%                       |
| CIFAR10 MobileNetV2 10epoch | Quantization 4bits                      | [16,24,56,56]                     | 0.10s          | 89.42%                       |
| CIFAR10 MobileNetV2 10epoch | Sort Quantization 4bits(4splits,2bits)  | [16,24,56,56]                     | 0.10s          | 93.38%                       |
| CIFAR10 MobileNetV2 10epoch | None                                    | [16,24,56,56]                     | 0.07s          | 94.21%                       |
| RTE Roberta-base 20epochs   | Sota                                    | [8,128,786]                       |                | 78.9%                        |
| RTE Roberta-base 20epochs   | None                                    | [8,128,786]\(the first layer)     | 0.39s          | 78.5% ~ 0.1%                 |
| RTE Roberta-base 20epochs   | Quantization 6bits                      | [8,128,786]\(the first layer)     | 0.39s          | 77.5%~0.1%                   |
| RTE Roberta-base 20epochs   | Sort Quantization 6bits(3bits,8splits)  | [8,128,786]\(the first layer)     | 0.39s          | 78.1%~0.2%                   |
| RTE Roberta-base 20epochs   | K-meas 6bits(50 iter)                   | [8,128,786]\(the first layer)     | 3.06s          | 52.7%,53.6%（the bug occurs) |
| RTE Roberta-base 20epochs   | K-meas 6bits(100 iter)                  | [8,128,786]\(the first layer)     | 5.27s          | 55.1%(the bug occurs)        |
| RTE Roberta-base 20epochs   | K-meas 6bits(100 iter)                  | [8,128,786]\(the sender layer)    | 5.27s          | 73.6%(TD)                    |
| RTE Roberta-base 20epochs   | Quantization 6bits                      | [8,128,786]\(the sender layer)    | 0.40s          | 73.1%(TD)                    |
| RTE Roberta-base 20epochs   | Sort Quantization 6bits(3bits,8splits)  | [8,128,786]\(the sender layer)    | 0.40s          | 77.2%~0.9%(TD)               |
| RTE Roberta-base 20epochs   | K-meas 6bits(50 iter)                   | [8,128,786]\(the  last two layer) | 3.05s          | 79.4%                        |
| RTE Roberta-base 20epochs   | Quantization 6bits                      | [8,128,786]\(the  last two layer) | 0.4s           | 52.2%                        |
| RTE Roberta-base 20epochs   | Sort Quantization 6bits(3bits, 8splits) | [8,128,786]\(the  last two layer) | 0.4s           | 75.0%                        |
| Cola Roberta-base 20epochs  | Sota                                    | [8,128,786]                       |                | 0.636(Matthew) 85.0%(acc)    |
| Cola Roberta-base 20epochs  | K-meas 6bits(100 iter)                  | [8,128,786]\(the  last two layer) | 1.32s          | 0.633～0.006(Matthew)        |
| Cola Roberta-base 20epochs  | Sort Quantization 6bits(3bits, 8splits) | [8,128,786]\(the  last two layer) | 0.4s           | 0.591～0.006(Matthew)        |
| Cola Roberta-base 20epochs  | Quantization 6bits                      | [8,128,786]\(the  last two layer) | 0.4s           | 0.587 ～0.007(Matthew)       |



# Ablation study



| Settings            | Method                                                       | Input size                            | Time per batch | Acc     |
| ------------------- | ------------------------------------------------------------ | ------------------------------------- | -------------- | ------- |
| CIFAR10 MobilenetV2 | Sort Quantization 4bits(3bits 2split)                        | [256,32,112,112]\(first one last one) | 0.40           | 87.1%   |
| CIFAR10 MobilenetV2 | Sort Quantization 8bits(6bits 4split)                        | [256,32,112,112]\(first one last one) | 0.40           | 95.5%   |
| CIFAR10 MobilenetV2 | Sort Quantization 12bits(9bits 8split)                       | [256,32,112,112]\(first one last one) | 0.44           | 95.9%   |
| CIFAR10 MobilenetV2 | Sort Quantization 16bits(12bits 16split)                     | [256,32,112,112]\(first one last one) | 0.49           | 96.1%   |
| CIFAR10 MobilenetV2 | Sort Quantization 8bits(40 epochs)+Sort Quantization 12bits(40 epochs) | [256,32,112,112]\(first one last one) | 0.42           | 95.72%  |
| Finetune            | Quantization 10bits                                          |                                       |                | 68%     |
| Finetune            | Quantization 11bits                                          |                                       |                | 91% 10% |
| Finetune            | Quantization 12bits                                          |                                       |                | 95.4%   |
| Finetune            | Quantization 16bits                                          |                                       |                | 96.0%   |
| RTE Roberta         | Sort Quantization 4bits(9bits 8split)                        | firsrt 1 last 2                       |                | 52.2%   |
| RTE Roberta         | Sort Quantization 8bits(6bits 4split)                        | firsrt 1 last 2                       |                | 79.7%   |
| RTE Roberta         | Sort Quantization 12bits(9bits 8split)                       | firsrt 1 last 2                       |                | 79.7%   |
| RTE Roberta         | Sort Quantization 16bits(12bits 16split)                     | firsrt 1 last 2                       |                | 79.7%   |
|                     |                                                              |                                       |                |         |



## TEST CODE

```
./tests/dataparallel_test_cv.py
```

