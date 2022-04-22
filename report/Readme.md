---
typora-copy-images-to: ./pic
---



# 1.distgpipe training

On CIFAR10 MobilenetV2 training

Traditional quantization is one step, one min per tensor. Multiple quantization has multiple steps and mins

Here are some important data. And I have done some train efficiency tests on Gpipe, which speed up the process by at least 30% compared to naive model parallelism training.

## 1.1 CIFAR10

I have tested CIFAR10 on MobileNet with several compression method.

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

## 1.2 RTE

RTE with Roberta backend.

| Training method | Compression method             | Acc%       |
| --------------- | ------------------------------ | ---------- |
| Finetune        | No                             | 79.7%~0.2% |
| Finetune        | Quant6                         | 75.1%~3.2% |
| Finetune        | Sort Quant6(4bits 2bits split) | 77.4%~2.0% |
| Finetune        | Quant8                         | 78.4%~1%   |
| Finetune        | Sort Quant8(6bits 2bits split) | 78.4%~1%   |



## 1.2 Sort Quantization

Haokang_quantization

Here is the pseudocode

![image-20220419152830075](./pic/image-20220419152830075.png)



# Ablation study



![image-20220415172007987](./pic/image-20220415172007987.png)

![image-20220415173956248](./pic/image-20220415173956248.png)



# Altogether Ablation Study

## CIFAR10

Backend:MobileNetV2

Client Server Partition: First and last layer

| Batchsize     | Activation Memory Size(al together) | Compression Method(default3:1) | Compression Ratio | Validation Acc | Bandwidth |
| ------------- | ----------------------------------- | ------------------------------ | ----------------- | -------------- | --------- |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 16bits       | 0.5               | 96.0%±0.13%    |           |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 12bits       | 0.375             | 95.9%±0.14%    |           |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 8bits        | 0.25              | 95.7%±0.03%    |           |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 4bits        | 0.125             | 87.10%         |           |

## CIFAR100

Backend:MobileNetV2

Client Server Partition: First and last layer

Since the activation memory size is the same as the CIFAR10 dataset, the bandwidth is the same as the bandwidth in CIFAR10

| Batchsize     | Activation Memory Size(al together) | Compression Method(default3:1) | Compression Ratio | Validation Acc |
| ------------- | ----------------------------------- | ------------------------------ | ----------------- | -------------- |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | No                             | 1                 | 80.92%         |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 16bits       | 0.5               | 80.85%         |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 12bits       | 0.375             | 80.61%         |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 8bits        | 0.25              | 78.83%         |

## FOOD101

Backend:MobileNetV2

Client Server Partition: First and last layer

Since the activation memory size is the same as the CIFAR10 dataset, the bandwidth is the same as the bandwidth in CIFAR10

| Batchsize     | Activation Memory Size(al together) | Compression Method(default3:1) | Compression Ratio | Validation Acc |
| ------------- | ----------------------------------- | ------------------------------ | ----------------- | -------------- |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | No                             | 1                 | 83.76%         |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 16bits       | 0.5               | 83.77%         |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 12bits       | 0.375             | 83.72%         |
| 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 8bits        | 0.25              |                |

## RTE

Backend:Roberta-base

Client Server Partition: First two and last two layers

| Batchsize    | activation memory size(al together) | Compression method(default3:1) | compression ratio | Validation acc(in cola is Matthew) | Bandwidth |
| ------------ | ----------------------------------- | ------------------------------ | ----------------- | ---------------------------------- | --------- |
| 32(4 chunks) | [32,128,768],[32,128,768]           | Sort Quantization 16bits       | 0.5               | 79.6%±0.18%                        |           |
| 32(4 chunks) | [32,128,768],[32,128,768]           | Sort Quantization 12bits       | 0.375             | 79.6%±0.20%                        |           |
| 32(4 chunks) | [32,128,768],[32,128,768]           | Sort Quantization 8bits        | 0.25              | 79.4%±0.21%                        |           |
| 32(4 chunks) | [32,128,768],[32,128,768]           | Sort Quantization 4bits        | 0.125             | 52.2%                              |           |

## COLA

Backend:Roberta-base

Client Server Partition: First two and last two layers

| Batchsize    | Activation Memory Size(Al together) | Compression Method(default3:1) | Compression Ratio | Matthew's Corelation |
| ------------ | ----------------------------------- | ------------------------------ | ----------------- | -------------------- |
| 32(4 chunks) | [32,128,768],[32,128,768]           | Sort Quantization 16bits       | 0.5               | 64.5±0.48            |
| 32(4 chunks) | [32,128,768],[32,128,768]           | Sort Quantization 12bits       | 0.375             | 63.93±0.22           |
| 32(4 chunks) | [32,128,768],[32,128,768]           | Sort Quantization 8bits        | 0.25              | 63.20±0.12           |
| 32(4 chunks) | [32,128,768],[32,128,768]           | Sort Quantization 4bits        | 0.125             | 0                    |







| Dataset  | Backend     | Batchsize     | activation memory size(al together) | Compression method(default3:1) | compression ratio | Validation acc(in cola is Matthew) | Bandwidth          |
| -------- | ----------- | ------------- | ----------------------------------- | ------------------------------ | ----------------- | ---------------------------------- | ------------------ |
| CIFAR10  | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]f1l1 | Sort Quantization 16bits       | 0.5               | 96.0%±0.13%                        | 160.73G/s 25.94G/s |
| CIFAR10  | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 12bits       | 0.375             | 95.9%±0.14%                        | 131.41G/s 17.97G/s |
| CIFAR10  | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 8bits        | 0.25              | 95.7%±0.03%                        | 89.51G/s 13.03G/s  |
| CIFAR10  | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 4bits        | 0.125             | 87.10%                             | 37.13G/s 6.51G/s   |
| CIFAR100 | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | No                             | 1                 | 80.92%                             |                    |
| CIFAR100 | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 16bits       | 0.5               | 80.85%                             |                    |
| CIFAR100 | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 12bits       | 0.375             | 80.61%                             |                    |
| CIFAR100 | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 8bits        | 0.25              | 78.83%                             |                    |
| CIFAR100 | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 8bits(1:1)   | 0.25              | 80.52%                             |                    |
| Food101  | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | No                             | 1                 | 83.76%                             |                    |
| Food101  | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 16bits       | 0.5               | 83.77%                             |                    |
| Food101  | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 12bits       | 0.375             | 83.72%                             |                    |
| Food101  | MobileNetV2 | 256(8 chunks) | [256,32,112,112] [256,1280,7,7]     | Sort Quantization 8bits        | 0.25              |                                    |                    |
| RTE      | Roberta     | 32(4 chunks)  | [32,128,768],[32,128,768]f2l2       | Sort Quantization 16bits       | 0.5               | 79.6%±0.18%                        | 11.04G/s           |
| RTE      | Roberta     | 32(4 chunks)  | [32,128,768],[32,128,768]           | Sort Quantization 12bits       | 0.375             | 79.6%±0.20%                        | 8.19G/s            |
| RTE      | Roberta     | 32(4 chunks)  | [32,128,768],[32,128,768]           | Sort Quantization 8bits        | 0.25              | 79.4%±0.21%                        | 5.37G/s            |
| RTE      | Roberta     | 32(4 chunks)  | [32,128,768],[32,128,768]           | Sort Quantization 4bits        | 0.125             | 52.2%                              | 2.774G/s           |
| Cola     | Roberta     | 32(4 chunks)  | [32,128,768],[32,128,768]f2l2       | Sort Quantization 16bits       | 0.5               | 64.5±0.48                          | 11.33G/s           |
| Cola     | Roberta     | 32(4 chunks)  | [32,128,768],[32,128,768]           | Sort Quantization 12bits       | 0.375             | 63.93±0.22                         | 7.96G/s            |
| Cola     | Roberta     | 32(4 chunks)  | [32,128,768],[32,128,768]           | Sort Quantization 8bits        | 0.25              | 63.20±0.12                         | 5.91G/s            |
| Cola     | Roberta     | 32(4 chunks)  | [32,128,768],[32,128,768]           | Sort Quantization 4bits        | 0.125             | 0                                  | 2.65G/s            |

**Bandiwidth** is calculated by recv_bytes / recv_time

Also, bandwidth has a linear relationship with recv size.TEST CODE

```
python test_vision_dgpipe.py --sortquant --quant <quant bit> --split <split bit> --log <logdir> --chunk <chunk>
python test_nlp_dgpipe.py --sortquant --quant <quant bit> --split <split bit> --log <logdir> --chunk <chunk>
```

  for bandwidth detection, check 

```
./test_cv_bandwidth.py --bandwidth
./test_nlp_bandwidth.py --bandwidth
```

| Method                                  | Separate Strategy                 | Acc                    |
| --------------------------------------- | --------------------------------- | ---------------------- |
| Sota                                    | None                              | 78.9%                  |
| None                                    | first two layers last layer       | 78.5% ~ 0.1%           |
| Quantization 6bits                      | first two layers last layer       | 77.5%~0.1%             |
| Sort Quantization 6bits(3bits,8splits)  | [8,128,786]\(the first layer)     | 78.1%~0.2%             |
| K-meas 6bits(50 iter)                   | first two layers last layer       | 52.7%,53.6%            |
| K-meas 6bits(100 iter)                  | first two layers last layer       | 55.1%                  |
| K-meas 6bits(50 iter)                   | first two layers last two layer   | 79.4%                  |
| Quantization 6bits                      | [8,128,786]\(the  last two layer) | 52.2%                  |
| Sort Quantization 6bits(3bits, 8splits) | [8,128,786]\(the  last two layer) | 75.0%                  |
| Sota                                    | [8,128,786]                       | 0.636(Matthew)         |
| K-meas 6bits(100 iter)                  | [8,128,786]\(the  last two layer) | 0.633～0.006(Matthew)  |
| Sort Quantization 6bits(3bits, 8splits) | [8,128,786]\(the  last two layer) | 0.591～0.006(Matthew)  |
| Quantization 6bits                      | [8,128,786]\(the  last two layer) | 0.587 ～0.007(Matthew) |
