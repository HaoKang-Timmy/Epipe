# speed and acc

Here is a test of model parallel + parallelism pipeline and data-parallel

You could use the code provided at

https://github.com/timmywanttolearn/gpipe_test

to see the speed and acc of model parallel + parallelism pipeline.

You can see the data-parallel code here.

# Result

#  parallel pipeline vs data-parallel

| Experiment        | Dataset | Backend     | GPUs | Batch size    | Learning rate | Top-1 acc (%) | Throughput | Speed up |
| ----------------- | ------- | ----------- | ---- | ------------- | ------------- | ------------- | ---------- | -------- |
| Pipeline-2gpu     | CIFAR10 | MobilenetV2 | 2    | 64(4 chunks)  | 0.005         | 95.89±0.07    | 228.57/s   | 0.607×   |
| Dataparallel-2gpu | CIFAR10 | MobilenetV2 | 2    | 64            | 0.005         | 95.83±0.04    | 376.47/s   | 1×       |
| Pipeline-4gpu     | CIFAR10 | MobilenetV2 | 4    | 256(4 chunks) | 0.02          | 96.03±0.14    | 400.30/s   | 1.07×    |
| Pipeline-4gpu     | CIFAR10 | MobilenetV2 | 4    | 256(8 chunks) | 0.02          | 96.07±0.05    | 397.30/s   | 1.06×    |
| Dataparallel-4gpu | CIFAR10 | MobilenetV2 | 4    | 256           | 0.02          | 95.94±0.09    | 627.22/s   | 1.66×    |
| Pipeline-2gpu     | RTE     | Roberta     | 2    | 32(4 chunks)  | 2e-5          | 78.59±0.21    | 61.53/s    | 0.80×    |
| Pipeline-2gpu     | RTE     | Roberta     | 2    | 64(4 chunks)  | 4e-5          | 77.56±0.39    | 68.82/s    | 0.90×    |
| Dataparallel-2gpu | RTE     | Roberta     | 2    | 32            | 2e-5          | 79.0±0.27     | 76.19/s    | 1×       |
| Pipeline-4gpu     | RTE     | Roberta     | 4    | 64(4 chunks)  | 4e-5          | 78.17±0.44    | 106.40/s   | 1.40×    |
| Pipeline-4gpu     | RTE     | Roberta     | 4    | 64(2 chunks)  | 4e-5          | 78.15±0.22    | 96.40/s    | 1.27×    |
| Dataparallel-4gpu | RTE     | Roberta     | 4    | 64            | 4e-5          | 78.4±0.21     | 95.53/s    | 1.25×    |

You could see that nlp model performs better at model parallel. This is because I only put first and last layer at the client gpu.

CV models are always hard to separate.

If you separate the model averagely, it changes

| Experiment    | Dataset | Backend     | GPUs | Batch size   | Learning rate | Throughput | Speed up |
| ------------- | ------- | ----------- | ---- | ------------ | ------------- | ---------- | -------- |
| Pipeline-2gpu | CIFAR10 | MobilenetV2 | 2    | 64(4 chunks) | 0.005         | 318.74/s   | 0.847×   |

And also NLP models are slow in data-parallel mode

| Model   | Train-method         | number of gpu | Total batchsize | Throughput |
| ------- | -------------------- | ------------- | --------------- | ---------- |
| Roberta | Dataparallel         | 2             | 32              | 76.19/s    |
| Roberta | Dataparallel         | 3             | 48              | 90.56/s    |
| Roberta | Dataparallel         | 4             | 64              | 94.12/s    |
| Roberta | Dataparallel         | 2             | 64              | 98.46/s    |
| Roberta | Dataparallel         | 3             | 96              | 111.67/s   |
| Roberta | Dataparallel         | 4             | 128             | 121.93/s   |
| Roberta | Parallelism pipeline | 2(chunk2)     | 32              | 64.03/s    |
| Roberta | Parallelism pipeline | 2(chunk4)     | 32              | 62.19/s    |
| Roberta | Parallelism pipeline | 2(chunk4)     | 64              | 69.57/s    |

How come that dataparallel in Roberta so slow

| Train method | number of gpu | batch size per GPU | time per batch | Data transfer time | Backward time |
| ------------ | ------------- | ------------------ | -------------- | ------------------ | ------------- |
| Dataparallel | 2             | 16                 | 0.401          | 0.037              | 0.244         |
| Dataparallel | 3             | 16                 | 0.519          | 0.100              | 0.277         |
| Dataparallel | 4             | 16                 | 0.657          | 0.161              | 0.310         |

In MobileNetV2

| Train method | number of GPU | batch size per GPU | time per batch | Data transfer time | Backward time |
| ------------ | ------------- | ------------------ | -------------- | ------------------ | ------------- |
| Dataparallel | 2             | 64                 | 0.341          | 0.064              | 0.243         |
| Dataparallel | 3             | 64                 | 0.364          | 0.098              | 0.241         |
| Dataparallel | 4             | 64                 | 0.389          | 0.13               | 0.240         |

Is it because the size of the model?

| Model        | Parameters    |
| ------------ | ------------- |
| MobileNetV2  | 2~ milion     |
| Roberta base | 123~ milion   |
| VGG 13       | 133.05 milion |
| AMoebanet    | 3~ milion     |
| Resnet101    | 44milion      |

For VGG13

| Train method | number of GPU | batch size per GPU | time per batch | Data transfer time | Backward time |
| ------------ | ------------- | ------------------ | -------------- | ------------------ | ------------- |
| Dataparallel | 2             | 64                 | 0.704          | 0.064              | 0.636         |
| Dataparallel | 4             | 64                 | 0.779          | 0.135              | 0.640         |

Seems not.

The true reason is the optimizer the model uses.

If I changed AdamW to **SGD** for Roberta, the reason is shown below

| Train method  | number of GPU | batch size per GPU | time per batch | Data transfer time | Backward time |
| ------------- | ------------- | ------------------ | -------------- | ------------------ | ------------- |
| Data parallel | 2             | 16                 | 0.252          | 0.039              | 0.097         |
| Data parallel | 4             | 16                 | 0.435          | 0.171              | 0.091         |

But **SGD** performs really bad at Roberta Model

To double-check my statement, I use **AdamW** to train MobileNetV2.

| Train method | number of GPU | batch size per GPU | time per batch | Data transfer time | Backward time |
| ------------ | ------------- | ------------------ | -------------- | ------------------ | ------------- |
| Dataparallel | 2             | 64                 | 0.323          | 0.064              | 0.246         |
| Dataparallel | 4             | 16                 | 0.397          | 0.132              | 0.248         |

Still not right, is it because of the parameters of the model? Lets try bigger one VGG13

| Train method | number of GPU | batchsize per GPU | time per batch | Data transfer time | Backward time |
| ------------ | ------------- | ----------------- | -------------- | ------------------ | ------------- |
| Dataparallel | 2             | 64                | 0.730          | 0.064              | 0.664         |
| Dataparallel | 4             | 64                | 0.817          | 0.142              | 0.674         |

# Reproduce

```
python3 ./CIFAR10_nGPU.py --worker 2 --log-dir ./test.txt
python3 ./NLP_nGPU.py --worker 2 --log-dir ./test.txt
```

