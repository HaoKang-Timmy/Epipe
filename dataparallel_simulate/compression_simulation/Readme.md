# Compression algorithm simluate

## Result of quantization ,sort quantization, and pruning

MobilenetV2 CIFAR10

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

NLP tasks with Roberta

| Tasks | Training method | Compression method                | Validation_value |
| ----- | --------------- | --------------------------------- | ---------------- |
| Cola  | Sota            | None                              | 0.636            |
| Cola  | Finetune        | None                              | 0.634～0.008     |
| Cola  | Finetune        | Prune 0.5                         | 0.633～0.013     |
| Cola  | Finetune        | Quantization 16                   | 0.632～0.010     |
| Cola  | Finetune        | Quantization 10                   | 0.635~0.014      |
| Cola  | Finetune        | Quantization 8                    | 0.644~0.001      |
| Cola  | Finetune        | Quantization 4                    | 0(acc: 69.1%)    |
| RTE   | Sota            | None                              | 78.9%            |
| RTE   | Finetune        | None                              | 78.4% ~ 0.6%     |
| RTE   | Finetune        | Prune 0.5                         | 79.3%~ 0.7%      |
| RTE   | Finetune        | Quantization 16                   | 78.7% ~ 0.7%     |
| RTE   | Finetune        | Quantization 10                   | 0.783~1.1%       |
| RTE   | Finetune        | Quantization 8                    | 77.5% ~ 0.8%     |
| RTE   | Finetune        | Sort Quantization 6bits( 4splits) | 79.5% ~0.5%      |
| RTE   | Finetune        | Quantization 4                    | 52.2% ~0.1%      |

## Ablation Study of sort quantization

| Settings            | Method                                                       | Input size                            | Time per batch | Acc     |
| ------------------- | ------------------------------------------------------------ | ------------------------------------- | -------------- | ------- |
| CIFAR10 MobilenetV2 | Sort Quantization 4bits(3bits 2split)                        | [256,32,112,112]\(first one last one) | 0.40           | 87.1%   |
| CIFAR10 MobilenetV2 | Sort Quantization 8bits(6bits 4split)                        | [256,32,112,112]\(first one last one) | 0.40           | 95.7%   |
| CIFAR10 MobilenetV2 | Sort Quantization 12bits(9bits 8split)                       | [256,32,112,112]\(first one last one) | 0.44           | 95.9%   |
| CIFAR10 MobilenetV2 | Sort Quantization 16bits(12bits 16split)                     | [256,32,112,112]\(first one last one) | 0.49           | 96.1%   |
| CIFAR10 MobilenetV2 | Sort Quantization 8bits(40 epochs)+Sort Quantization 12bits(40 epochs) | [256,32,112,112]\(first one last one) | 0.42           | 95.72%  |
| Finetune            | Quantization 10bits                                          |                                       |                | 68%     |
| Finetune            | Quantization 11bits                                          |                                       |                | 91% 10% |
| Finetune            | Quantization 12bits                                          |                                       |                | 95.4%   |
| Finetune            | Quantization 16bits                                          |                                       |                | 96.0%   |
| RTE Roberta         | Sort Quantization 4bits(2bits 2split)                        | firsrt 1 last 2                       |                | 52.2%   |
| RTE Roberta         | Sort Quantization 8bits(6bits 2split)                        | firsrt 1 last 2                       |                | 79.7%   |
| RTE Roberta         | Sort Quantization 12bits(9bits 3split)                       | firsrt 1 last 2                       |                | 79.7%   |
| RTE Roberta         | Sort Quantization 16bits(12bits 4split)                      | firsrt 1 last 2                       |                |         |

## Comparing with k-means

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
| RTE Roberta-base 20epochs   | K-meas 6bits(50 iter)                   | [8,128,786]\(the  last two layer) | 3.05s          | 79.4%                        |
| RTE Roberta-base 20epochs   | Quantization 6bits                      | [8,128,786]\(the  last two layer) | 0.4s           | 52.2%                        |
| RTE Roberta-base 20epochs   | Sort Quantization 6bits(3bits, 8splits) | [8,128,786]\(the  last two layer) | 0.4s           | 75.0%                        |
| Cola Roberta-base 20epochs  | Sota                                    | [8,128,786]                       |                | 0.636(Matthew) 85.0%(acc)    |
| Cola Roberta-base 20epochs  | K-meas 6bits(100 iter)                  | [8,128,786]\(the  last two layer) | 1.32s          | 0.633～0.006(Matthew)        |
| Cola Roberta-base 20epochs  | Sort Quantization 6bits(3bits, 8splits) | [8,128,786]\(the  last two layer) | 0.4s           | 0.591～0.006(Matthew)        |
| Cola Roberta-base 20epochs  | Quantization 6bits                      | [8,128,786]\(the  last two layer) | 0.4s           | 0.587 ～0.007(Matthew)       |

# Reproduce

```
python3 datdaparallel_test_cv.py --root <dataset root> --log-dir ./test.txt --quant <quantization bits> --prune <prune ratio> --kmeans <use or not> --multi <whether to use sort quant>
python3 datdaparallel_test_nlp.py --root <dataset root> --log-dir ./test.txt --quant <quantization bits> --prune <prune ratio> --kmeans <use or not> --multi <whether to use sort quant> --task rte
```

