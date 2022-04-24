# Speed and Performance

Here is a test of model parallel + parallelism pipeline and data-parallelism

You could use the code provided at

https://github.com/timmywanttolearn/gpipe_test

to see the speed and acc of model parallel + parallelism pipeline.

You can see the data-parallel code here.

# 1.Result

#  1.1Parallelism Pipeline vs Data-parallel

| Experiment                 | Dataset | Backend     | GPUs | Batch size    | Learning rate | Top-1 acc (%) | Throughput | Speed up |
| -------------------------- | ------- | ----------- | ---- | ------------- | ------------- | ------------- | ---------- | -------- |
| Pipeline-2gpu              | CIFAR10 | MobilenetV2 | 2    | 64(4 chunks)  | 0.005         | 95.89±0.07    | 228.57/s   | 0.607×   |
| Pipeline-2gpu(origin code) | CIFAR10 | MobilenetV2 | 2    | 64(4 chunks)  | 0.005         | None          | 213.33/s   | 0.566×   |
| Dataparallel-2gpu          | CIFAR10 | MobilenetV2 | 2    | 64            | 0.005         | 95.83±0.04    | 376.47/s   | 1×       |
| Pipeline-4gpu              | CIFAR10 | MobilenetV2 | 4    | 256(4 chunks) | 0.02          | 96.03±0.14    | 400.30/s   | 1.07×    |
| Pipeline-4gpu(origin code) | CIFAR10 | MobilenetV2 | 4    | 256(4 chunks) | 0.005         | None          | 419.67/s   | 1.11×    |
| Pipeline-4gpu              | CIFAR10 | MobilenetV2 | 4    | 256(8 chunks) | 0.02          | 96.07±0.05    | 397.30/s   | 1.06×    |
| Dataparallel-4gpu          | CIFAR10 | MobilenetV2 | 4    | 256           | 0.02          | 95.94±0.09    | 627.22/s   | 1.66×    |
| Pipeline-2gpu              | RTE     | Roberta     | 2    | 32(4 chunks)  | 2e-5          | 78.59±0.21    | 61.53/s    | 0.80×    |
| Pipeline-2gpu              | RTE     | Roberta     | 2    | 64(4 chunks)  | 4e-5          | 77.56±0.39    | 68.82/s    | 0.90×    |
| Dataparallel-2gpu          | RTE     | Roberta     | 2    | 32            | 2e-5          | 79.0±0.27     | 76.19/s    | 1×       |
| Pipeline-4gpu              | RTE     | Roberta     | 4    | 64(4 chunks)  | 4e-5          | 78.17±0.44    | 106.40/s   | 1.40×    |
| Pipeline-4gpu              | RTE     | Roberta     | 4    | 64(2 chunks)  | 4e-5          | 78.15±0.22    | 96.40/s    | 1.27×    |
| Dataparallel-4gpu          | RTE     | Roberta     | 4    | 64            | 4e-5          | 78.4±0.21     | 95.53/s    | 1.25×    |

You could see that nlp model performs better at model parallel. This is because I only put first and last layer at the client gpu.

CV models are always hard to separate.

If you separate the model averagely, it changes

| Experiment    | Dataset | Backend     | GPUs | Batch size   | Learning rate | Throughput | Speed up |
| ------------- | ------- | ----------- | ---- | ------------ | ------------- | ---------- | -------- |
| Pipeline-2gpu | CIFAR10 | MobilenetV2 | 2    | 64(4 chunks) | 0.005         | 318.74/s   | 0.847×   |

# 1.2Reproduce

```
bash ./test.sh
```

