# speed and acc

Here is a test of model parallel + parallelism pipeline and data parallel

You could use the code provided at

https://github.com/timmywanttolearn/gpipe_test

to see the speed and acc of model parallel + parallelism pipeline.

You can see the data parallel code here.

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
| Pipeline-2gpu     | RTE     | Roberta     | 2    | 64(4 chunks)  | 4e-5          | 76.56±0.39    | 68.82/s    | 0.90×    |
| Dataparallel-2gpu | RTE     | Roberta     | 2    | 32            | 2e-5          | 79.0±0.27     | 76.19/s    | 1×       |
| Pipeline-4gpu     | RTE     | Roberta     | 4    | 64(4 chunks)  | 4e-5          | 78.17±0.44    | 106.40/s   | 1.40×    |
| Pipeline-4gpu     | RTE     | Roberta     | 4    | 64(2 chunks)  | 4e-5          | 78.15±0.22    | 96.40/s    | 1.01×    |
| Dataparallel-4gpu | RTE     | Roberta     | 4    | 64            | 4e-5          | 77.4±0.21     | 95.53/s    | 1.25×    |

You could see that nlp model performs better at model parallel. This is because I only put first and last layer at the client gpu.

CV models are always hard to separate.

If you separate the model averagely, it changes

| Experiment    | Dataset | Backend     | GPUs | Batch size   | Learning rate | Throughput | Speed up |
| ------------- | ------- | ----------- | ---- | ------------ | ------------- | ---------- | -------- |
| Pipeline-2gpu | CIFAR10 | MobilenetV2 | 2    | 64(4 chunks) | 0.005         | 228.57/s   | 0.851×   |

And also NLP models are slow at data parallel mode

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

| Train method | number of gpu | batchsize per gpu | time per batch | Data transfer time | Backward time |
| ------------ | ------------- | ----------------- | -------------- | ------------------ | ------------- |
| Dataparallel | 2             | 16                | 0.417          | 0.037              | 0.244         |
| Dataparallel | 3             | 16                | 0.537          | 0.100              | 0.307         |
| Dataparallel | 4             | 16                | 0.657          | 0.161              | 0.378         |

In MobileNetV2

| Train method | number of gpu | batchsize per gpu | time per batch | Data transfer time | Backward time |
| ------------ | ------------- | ----------------- | -------------- | ------------------ | ------------- |
| Dataparallel | 2             | 64                | 0.341          | 0.064              | 0.243         |
| Dataparallel | 3             | 64                | 0.364          | 0.098              | 0.241         |
| Dataparallel | 4             | 64                | 0.389          | 0.13               | 0.240         |

# Reproduce

```
python3 ./CIFAR10_nGPU.py --worker 2 --log-dir ./test.txt
python3 ./NLP_nGPU.py --worker 2 --log-dir ./test.txt
```

