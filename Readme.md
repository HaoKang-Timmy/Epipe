---

typora-copy-images-to: ./pic
---

# Test of different layers in edge

# Introductions and settings

I use gpipe to test different layers at the edge side.

## Settings

| Model             | MobileNetV2                                  |
| ----------------- | -------------------------------------------- |
| Dataset           | CIFAR10                                      |
| Training_strategy | train from scratch                           |
| lr_init           | 0.4                                          |
| Batch_size        | 1024                                         |
| Chunk             | 4(every batch is splited to 4 micro-batches) |
| Optimizer         | SGD                                          |
| Momentum          | 0.9                                          |
| Weight_decay      | 1e-4                                         |
| Epochs            | 200                                          |
| Scheduler         | cosannealing with linear warp up(20 epochs)  |
| Pruning methods   | pruning0.1                                   |

## Results

| layers at edge side     | val_acc1% |
| ----------------------- | --------- |
| first 1 + last 1 layers | 92.14     |
| first 2 +last 1 layers  | 92.22     |
| first 3 +last 1 layers  | 92.63     |
| first 1 +last 2 layers  | 90.84     |
| first 2 +last 2 layers  | 91.69     |
| First3 + last 2 layers  | 92.59     |
| first 1 +last 3 layers  | 91.12     |
| first 2 + last3 layers  | 91.72     |
| first 3 +last 3 layers  | 92.85     |

It shows that the more front layers are at the edge, the more accuracy of the model gets. Also, when we put more posterior layers at the edge, the accuracy seems to get lower(since the posterior layers suffers compression two times).

## code



# About quantization

## Settings

The same as above.

## Result and discussion

Last week I found that in big learning rate conditions, quantization performs really bad.

![image-20220215102351169](./pic/image-20220215102351169.png)

The picture shows above. 

However, when I try to decrease the learning rate to the half of the origin one(0.2 for 1024 images per batch), this condition gets better.

![image-20220223002245350](./pic/image-20220223002245350.png)

Things get better.

