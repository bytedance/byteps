# BytePS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

BytePS is a high performance and general distributed training framework. It supports TensorFlow, Keras, PyTorch, and MXNet, and can run on either TCP or RDMA network.

BytePS outperforms existing open-sourced distributed training frameworks by a large margin. For example, on a popular public cloud and with the same number of GPUs, BytePS can *double the training speed* (see below), compared with [Horovod](https://github.com/horovod/horovod)+[NCCL](https://github.com/NVIDIA/nccl).

## Performance

For demonstration, we test two models: VGG16 (communication-intensive) and Resnet50 (computation-intensive).

We use Tesla V100 16GB GPUs and set batch size equal to 64. The machines are in fact VMs on a popular public cloud. Each machine has 8 V100 GPUs with NVLink-enabled. Machines are inter-connected with 20 Gbps TCP/IP network.

BytePS outperforms Horovod (NCCL) by 44% for Resnet50, and 100% for VGG16.

<img src="/docs/images/perf_tcp_vgg16.png" width="360" height="220"><img src="/docs/images/perf_tcp_resnet50.png" width="360" height="220">

You can reproduce the results using the Dockerfiles and example scripts we provide.

Evaluation on RDMA networks can be found at [performance.md](docs/performance.md).

## Goodbye MPI, Hello Cloud

How can BytePS outperform Horovod by so much? One of the main reasons is that BytePS is designed for cloud and shared clusters, and throws away MPI.

MPI was born in the HPC world and is good for a cluster built with homogeneous hardware and for running a single job. However, cloud (or in-house shared clusters) is different.

This leads us to rethink the best communication strategy, as explained in [here](docs/rationale.md). In short, BytePS only uses NCCL inside a machine, while re-implements the inter-machine communication.

BytePS also incorporates many acceleration techniques such as hierarchical strategy, pipelining, tensor partitioning, NUMA-aware local communication, priority-based scheduling, etc.

## Quick Start

Before using BytePS, we assume you have already installed one or more of the following frameworks: TensorFlow / Keras / PyTorch / MXNet. BytePS depends on CUDA and NCCL.

Clone BytePS with its third party dependency:

```
git clone --recurse-submodules https://github.com/bytedance/byteps
```

Then `cd` into your BytePS directory and install.
```
python setup.py install
```
Note: you may set `BYTEPS_USE_RDMA=1` to install with RDMA support.

We provide a [step-by-step tutorial](docs/step-by-step-tutorials.md) for you to run benchmark training tasks.

Also refer to [Documentations](docs) for how to [launch distributed jobs](docs/running.md) and more [detailed configurations](docs/env.md).

## Use BytePS in Your Code

Though being totally different at its core, BytePS is highly compatible with Horovod interfaces (Thank you, Horovod community!). We chose Horovod interfaces in order to minimize your efforts for testing BytePS.

If your tasks only rely on Horovod's allreduce and broadcast, you should be able to switch to BytePS in 1 minute. Simply replace `import horovod.tensorflow as hvd` by `import byteps.tensorflow as bps`, and then replace all `hvd` in your code by `bps`. If your code invokes `hvd.allreduce` directly, you should also replace it by `bps.push_pull`.

Many of our examples were copied from Horovod and modified in this way. For instance, compare the MNIST example for [BytePS](https://github.com/bytedance/byteps/blob/master/example/tensorflow/tensorflow_mnist.py) and [Horovod](https://github.com/horovod/horovod/blob/master/examples/tensorflow_mnist.py).

## Limitations and Future Plans

BytePS does not support pure CPU training for now. One reason is that some [assumptions](docs/rationale.md) of BytePS do not hold for CPU training. Consequently, you need CUDA and NCCL to build and run BytePS.

We would like to have below features, and it is not hard to implement them in BytePS architecture. However, they are not implemented yet:
* Sparse model training
* Asynchronous training
* Fault-tolerance
* Straggler-mitigation
