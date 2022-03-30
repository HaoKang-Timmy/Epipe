# TO DO LIST

## What I have done

### 1.distgpipe training

On CIFAR10 MobilenetV2 training

Traditional quantization is one step, one min per tensor. Multiple quantization has multiple steps and mins

| Training method         | Compression method          | Acc%   |
| ----------------------- | --------------------------- | ------ |
| tfs(train from scratch) | No                          | 94.07% |
| tfs(train from scratch) | Quantization16(traditional) | 93.94% |
| tfs(train from scratch) | Prune 0.5                   | 94.02% |
| Finetune                | No                          | 96.03% |
| Finetune                | Quantization16(traditional) | 96.07% |
| Finetune                | Prune 0.5                   | 96.27% |

Here are some important data. And I have done some train efficiency tests on dist-gpipe, which speed up the process at least 30%.

### Dataparallel tests

I have trained 100epochs here.

| Training method | Compression method                 | Acc%        |
| --------------- | ---------------------------------- | ----------- |
| Finetune        | No                                 | 95.9%       |
| Finetune        | Quantization 16bits                | 95.7%       |
| Finetune        | Prune0.5                           | 96.1%       |
| Finetune        | Quantization 11bits                | 91.3% 84.5% |
| Finetune        | Haokang_quantization 8bits 8splits | 95.47%      |

The reason that quantization 11bits has two acc is that, it's curve first climb quickly like quantization 16bits but suddenly fall to 60% and then climb slowly.

## To do

1. Finish multiple quantization in distributed method and run samples of CIFAR10 using MobileNetV2 backend.

   motivation: try this new method on the dist_gpipe system, test how many sets of steps and min are reasonable for **8bits** quantization

2. Fulfill the difference of input and output of quantization functions, record them and analyze.

   Motivation: analyze why the error will increase during small bits quantization training. And How to decrease it? Any specification that could represent the tensors distance that could give a response answer to acc decay dramatically.

3. Finish Roberta classification tasks using (prun ,quant,multiple quant) and analyze the results

   Motivation: tests compression layers on NLP tasks(don't prun nn.Embedding again).

## TEST CODE

```
./tests/dataparallel_test_cv.py
```

