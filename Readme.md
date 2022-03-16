# Training results

## settings

| Model              | MobileNetV2(tfs from kuangliu and fintune from torchvision) |
| ------------------ | ----------------------------------------------------------- |
| Dataset            | CIFAR10                                                     |
| Optimizer          | SGD with 0.9 momentum                                       |
| Scheduler          | Coslinear with 20 epochs of liner warmup                    |
| train method       | train from scratch and fintune                              |
| gpipe chunk        | 4 (which does not affect the accuracy)                      |
| compression method | Pruning,quantization,None                                   |
| pic size           | 224\*224(fintune) and 32\*32(tfs)                           |
| Epochs             | 100                                                         |
| Batch size         | 256(fintune) and 1024(tfs)                                  |

## Fintune

| compression                     | learning rate | Validation acc %         |
| ------------------------------- | ------------- | ------------------------ |
| quantization12(running)+prun0.8 | 0.001         | 91.93                    |
| quantization12(running)+prun0.8 | 0.01          | 79.45(with huge tremble) |
| quantization12(running)+prun0.8 | 0.005         | 95.13                    |
| quantization12(running)+prun0.6 | 0.001         | 92.10                    |
| quantization12(running)+prun0.6 | 0.01          | 95.58                    |
| quantization12(running)+prun0.4 | 0.01          | 95.65                    |
| quantization12(running)+prun0.4 | 0.001         | 91.31                    |
| No                              | 0.01          | 95.94                    |
| No                              | 0.001         | 92.54                    |

Quantization8 is still training.

## tfs

| compression     | learning rate | Validation acc % |
| --------------- | ------------- | ---------------- |
| Quant4(running) | 0.4           | 88.79            |
| Quant4          | 0.4           | 88.41            |
|                 |               |                  |

# How to run

Fintune full network with self-defined training log dir, pruning rate(could be ignored, then pruning layer won't be added to the network), and quantization bits. Also we can choose where is dataset by using --datasetdr 

```
python3 test.py --log-dir ./mygpipe_log/torchvision_mobilenet/pruning/quant8.txt --prun 0.6 --quant 8 --pretrained --lr 0.005 --datasetdr ./data
```

Train form scratch,with self define epochs, batch_size, whether warmup, quantization bits and pruning rate and learning rate

```
python3 test.py --log-dir ./mygpipe_log/mobilenet/quant8_1024_lr0.2.txt --quant 8 --lr 0.2 --epoches 100 --batches 1024 --warmup --datasetdr ./data
```

