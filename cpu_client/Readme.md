# CPU Training

A simulation by using CPUs to train client tasks and one GTX 1080 to train server tasks

## 1 CIFAR

## 1.1 Settings

| Backend     | Epochs | Lr    | Batch Size |
| ----------- | ------ | ----- | ---------- |
| MoblienetV2 | 40     | 0.005 | 64         |

## 1.2 Results

| Compression method | Sever Client Partition  | Throughputs | Validation Acc |
| ------------------ | ----------------------- | ----------- | -------------- |
| None               | First layer, last layer | 191.9/s     | 95.92          |
|                    |                         |             |                |
|                    |                         |             |                |

