---
typora-copy-images-to: ./pic
---

# Gpipe with compression tests

# Introductions and settings

Gpipe is a technique that built for model parallel. It separates batches to micro-batches and transfer them while computing.

Here is a test about some compression tests.

## Settings

| Model             | MobileNetV2                                  |
| ----------------- | -------------------------------------------- |
| Dataset           | CIFAR10                                      |
| Training_strategy | train from scratch                           |
| lr_init           | 0.4                                          |
| Batch_size        | 1024                                         |
| Chunk             | 4(every batch is splited to 4 micro batches) |
| Optimizer         | SGD                                          |
| Momentum          | 0.9                                          |
| Weight_decay      | 1e-4                                         |
| Epochs            | 200                                          |
| Scheduler         | cosannealing with linear warp up(20 epochs)  |
| Pruning methods   | topk pruning, downsample,quantization        |

## Results



pruning rate(input_pruning/input_before_pruning) vs validation_acc1

![image-20220215015840995](./pic/image-20220215015840995.png)



It shows that, when at same pruning rate(compression rate), using both pruning and quantization is better than any other choices provided.

Also, there are some interesting phenomenons.

First, quantization may increase accuracy when using pruning.In the graph we could see that when at pruning 0.1 and pruning 0.05, using quantization 8 and quantization 16 are better than origin pruning 0.1 or pruning 0.05. However, this only happens in the condition that pruning rate is high. For pruning 0.2 or pruning 0.4, it does not happen.



Second,only using quantization will cause huge overfit during training.



![image-20220215102351169](./pic/image-20220215102351169.png)

However, when using topk pruning together, the curve does not temble that much. 





# code

### quantization

https://github.com/timmywanttolearn/gpipe_test/blob/master/code/distributed_layers.py#:~:text=class%20QuantFunction(autograd.-,Function,-)%3A

### topk_pruning

https://github.com/timmywanttolearn/gpipe_test/blob/master/code/distributed_layers.py#:~:text=class-,TopkFunction,-(autograd.Function)%3A

### training code

https://github.com/timmywanttolearn/gpipe_test/blob/master/code/train.py#:~:text=for%20epoch%20in,file_save1.close()

# different layers

