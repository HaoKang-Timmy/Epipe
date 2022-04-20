# Comparing three kinds of compression algorithms

## Results

In this part, I generate tensors with `torch.rand` and test the harm of accuracy of three compression algorithms. Here are results.

The bits in Sort Quantization used for split and quantization are shown.

| Tensor Size | Compression Method           | MARE   |
| ----------- | ---------------------------- | ------ |
| 1000000     | Sort Quantization 8bits(6,2) | 0.0018 |
| 1000000     | Quantization 8bits           | 0.0035 |
| 1000000     | k-means                      | 0.0029 |

## Reproduce

See `./test.sh`

