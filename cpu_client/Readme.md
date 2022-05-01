---
typora-copy-images-to: ./pic
---

# CPU Training

A simulation by using CPUs to train client tasks and one GTX 1080 to train server tasks

## 1 CIFAR

## 1.1 Settings

| Backend     | Epochs | Lr    | Batch Size |
| ----------- | ------ | ----- | ---------- |
| MoblienetV2 | 40     | 0.005 | 64         |

## 1.2 Results

Here, since CPUs handle SVD faster than GPUs. I perform all PCA encode algorithms in CPUs.

| Hardware(Client,Server) | Compression method      | Chunk | Sever Client Partition  | Throughputs | Validation Acc |
| ----------------------- | ----------------------- | ----- | ----------------------- | ----------- | -------------- |
| CPU,CPU                 | None                    | None  |                         | 32.48/s     | 95.87          |
| Cpu,Gpu                 | None                    | 4     | First layer, last layer | 191.9/s     | 95.92          |
| GPU,GPU                 | None                    | 4     | First layer, last layer | 228.57/s    | 95.89          |
| Cpu,Gpu                 | Sort Quantization 8bits | 8     | First layer, last layer | 41.83/s     | 95.59          |
| Cpu,Gpu                 | PCA 12rank + PCA 2rank  | 8     | First layer, last layer | 40.50/s     |                |

# 2 Compression Algorithm Analyse

## 2.1 CPU

You can reproduce the results by executing `./CPUtest.py`

### Settings

| Activation Memory(Total/Batchsize) |
| ---------------------------------- |
| [32,112,112],[1280,7,7]            |

![image-20220501111122132](./pic/image-20220501111122132.png)

![image-20220501132419588](./pic/image-20220501132419588.png)



### Settings

| Activation Memory(Total/Batchsize) |
| ---------------------------------- |
| [128,768]                          |

![image-20220501132118695](./pic/image-20220501132118695.png)

![image-20220501132142973](./pic/image-20220501132142973.png)

## 2.2 GPU

| Activation Memory(Total/Batchsize) |
| ---------------------------------- |
| [32,112,112]                       |

![image-20220421173843653](./pic/test_gpu.jpg)

### Settings

| Activation Memory(Total/Batchsize) |
| ---------------------------------- |
| [128,768]                          |

![image-20220425174146944](./pic/image-20220425174146944.png)

