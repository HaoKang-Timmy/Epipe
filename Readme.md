# TO DO LIST

## What I have done

### 1.distgpipe training

On CIFAR10 MobilenetV2 training

Traditional quantization is one step, one min per tensor. Multiple quantization has multiple steps and mins

| Training method         | Compression method   | Acc%  |
| ----------------------- | -------------------- | ----- |
| tfs(train from scratch) | No                   | 94.07 |
| tfs(train from scratch) | Quant16(traditional) | 93.94 |
| tfs(train from scratch) | prun 0.5             | 94.02 |
| Fintune                 | No                   | 96.03 |
| Fintune                 | Quant16(traditional) | 96.07 |
| Fintune                 | Prun 0.5             | 96.27 |

Here are some important data. And I have done some train efficiency tests on dist-gpipe, which speed up the process at least 30%.

### Dataparallel tests

I have only trained 40epochs here. To test multiple quantization

| Training method | Compression method          | Acc%   |
| --------------- | --------------------------- | ------ |
| Fintune         | No                          | 92.70  |
| Fintune         | Quant16(traditional)        | 92.59% |
| Fintune         | prun0.5                     | 92.52% |
| Fintune         | Avgpool2d                   | 83.74% |
| Fintune         | Quant8(traditional)         | 62%    |
| Fintune         | Quant8 prun0.5(multiple 64) | 92.53% |

## To do

1. Finish multiple quantization in distributed method and run samples of CIFAR10 using MobileNetV2 backend.
2. Fulfill the difference of input and output of quantization functions, record them and analyse.
3. Finish Roberta classification tasks and analyse the results