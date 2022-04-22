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
| Dataparallel | 4             | 64                 | 0.684          | 0.169              | 0.640         |

Seems not.

The true reason is the optimizer the model uses.

If I changed AdamW to **SGD** for Roberta, the reason is shown below

| Train method  | number of GPU | batch size per GPU | time per batch | Data transfer time | Backward time |
| ------------- | ------------- | ------------------ | -------------- | ------------------ | ------------- |
| Data parallel | 2             | 16                 | 0.252          | 0.039              | 0.097         |
| Data parallel | 4             | 16                 | 0.435          | 0.171              | 0.091         |

Also there are tests for **Adam**

| Train method  | number of GPU | batch size per GPU | time per batch | Data transfer time | Backward time |
| ------------- | ------------- | ------------------ | -------------- | ------------------ | ------------- |
| Data parallel | 2             | 16                 | 0.397          | 0.045              | 0.254         |
| Data parallel | 4             | 16                 | 0.688          | 0.171              | 0.325         |

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

# 