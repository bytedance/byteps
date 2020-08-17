## Quick Start

First ensure that you have installed the newest version of BytePS (>=0.2.4). Currenly we only support MXNet. Please stay tuned for users of other frameworks.

If you are not famillar with gradient compression, I recommend you to read [gradient-compression-proposal](gradient-compression-proposal.md) first.

To enable gradient compression, you only need to pass a `dict` to `bps.DistributedTrainer`. Here is an example of using the scaled 1-bit compressor with error feedback and Nesterov's momentum.

```python
compression_params = {
  "compressor": "onebit",
  "ef": "vanilla",
  "momentum": "nesterov",
  "scaling": True
}

trainer = bps.DistributedTrainer(
  params, optimizer, optimizer_params, compression_params=compression_params)
```


## Best Practice of SoTA Algorithms

### dist-EF-SGDM

dist-EF-SGDM denotes the scaled 1-bit compressor with error feedback and Nesterov's momentum. 1-bit compressor extract sign bits and pack into compact data. Also, it will scale the sign bits with L1 norm to maintain the magnitude of the gradients. Refer to the [dist-EF-SGDM paper](https://arxiv.org/pdf/1905.10936.pdf) to get more details.

```python
compression_params = {
  "compressor": "onebit",
  "ef": "vanilla",
  "momentum": "nesterov",
  "scaling": True
}
```
### dithering

Dithering is a quantization algorithm which quantizes the floats into k integer levels. It use Enlias Encoding to packing those small integers. There are two kinds of dithering algorithms: linear(also called QSGD) and natural. Please refer to [qsgd paper](https://papers.nips.cc/paper/6768-qsgd-communication-efficient-sgd-via-gradient-quantization-and-encoding.pdf) and [natural dithering paper]() for more details.

```python
# linear dithering
compression_params = {
  "compressor": "dithering",
  "k": 2
}
```

```python
# natural dithering
compression_params = {
  "compressor": "dithering",
  "partition": "natural",
  "k": 2
}
```

### topk & randomk

topk and randomk are sparcification algorithms. Topk selects the k largest elements while randomk just randomly select k elements. In general, they should be used with error feedback to avoid divergence. Please refer to the [Sparsified SGD with memory](https://arxiv.org/pdf/1809.07599.pdf) to get more details.

```python
compression_params = {
  "compressor": "topk",
  "k": 0.001,
  "ef": "vanilla",
  "momentum": "nesterov"
}
```


```python
compression_params = {
  "compressor": "randomk",
  "k": 0.001,
  "ef": "vanilla",
  "momentum": "nesterov"
}
```

### FP16

We do not recommend using FP16 with other compressors. It will harm the performance because float16 has to converted to float32 when doing calucations, incurring large overhead. Hence we recommend you to use FP16 compression alone and it will give you a certain speedup. 

```python
compression_params = {
  "fp16": True
}
```


We have provided example scripts for reference. Please referr to `example/mxnet/gradient_compression`.

## List of Compression Parameters


| Keys | Description | Valid Input | 
| --- | --- | --- | 
| `compressor` | compression algorithms| "onebit" or "dithering" or "topk" or "randomk" |
| `k` | for sparcification methods such as topk and randomk, `k` determines how many elements are selected. For dithering, `k` means the quantization level. `k` must be specified when using dithering and topk and randomk. | a float from 0 to 1, indicating the proportion of how many gradients are selected, or a positive integer   |
| `ef` | error-feedback algorithms.   | "vanilla" |
| `momentum` |  momentum algorithms.  | "nesterov" |
| `fp16` | whether to enable fp16 conversion. | True or False |
| `scaling` | whether to enable scaling for onebit. | True or False |
| `seed` | random seed. | an positive integer |
| `normalize` | normalization method used in dithering. "max" is used if not specified. | "max" or "l2" |
| `partition` | partition method used in dithering. "linear" is used if not specified. | "linear" or "natural" |

For dithering, `max` usually gives better accuracy than `l2`. But `l2` provides more sparsity. So this is a trade-off. Also, we find that dithering with `l2`normalization and error feedback will diverge. We are investigating this problem.

## Environments 

There are some hyper-parameters that can be finetuned to obtain the best performance.

Compressing small gradidents might harm the performance because of the constant overhead of gradient compression. We support setting a threshold, and gradients below the threshold will not be compressed.
```sh
export BYTEPS_MIN_COMPRESS_BYTE=1024000
```

**Note that we do not recommend setting the threshold because it may hurt the accuracy. This is because some compression algorithms may require a different learning rate from the full-precision one's.**

Gradient compression jobs are pushed into a thread pool. You can modify the size of the thread pool.
```sh
export BYTEPS_THREADPOOL_SIZE=16
```