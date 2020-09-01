# BytePS

[![Build Status](https://travis-ci.org/bytedance/byteps.svg?branch=master)](https://travis-ci.org/bytedance/byteps)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Pypi](https://img.shields.io/pypi/v/byteps.svg)

BytePS is a high performance and general distributed training framework. It supports TensorFlow, Keras, PyTorch, and MXNet, and can run on either TCP or RDMA network.

BytePS outperforms existing open-sourced distributed training frameworks by a large margin. For example, on BERT-large training, BytePS can achieve ~90% scaling efficiency with 256 GPUs (see below), which is much higher than [Horovod](https://github.com/horovod/horovod)+[NCCL](https://github.com/NVIDIA/nccl). In certain scenarios, BytePS can double the training speed compared with Horovod+NCCL.

## News
- BytePS paper has been accepted to OSDI'20 (to appear soon). The code to reproduce the end-to-end evaluation is available [here](https://github.com/byteps/examples).
- Support [gradient compression](https://github.com/bytedance/byteps/pull/225).
- [v0.2.4](https://github.com/bytedance/byteps/tree/v0.2.4)
    * Fix compatibility issue with tf2 + standalone keras
    * Add support for tensorflow.keras
    * Improve robustness of broadcast
- [v0.2.3](https://github.com/bytedance/byteps/tree/v0.2.3)
    * Add DistributedDataParallel module for PyTorch
    * Fix the problem of different CPU tensor using the same name
    * Add skip_synchronize api for PyTorch
    * Add the option for lazy/non-lazy init
- [v0.2.0](https://github.com/bytedance/byteps/tree/v0.2)
    * Largely improve RDMA performance by enforcing page aligned memory.
    * Add IPC support for RDMA. Now support colocating servers and workers without sacrificing much performance.
    * Fix a hanging bug in BytePS server.
    * Fix RDMA-related segmentation fault problem during fork() (e.g., used by PyTorch data loader).
    * New feature: Enable mixing use of colocate and non-colocate servers, along with a smart tensor allocation strategy.
    * New feature: Add ``bpslaunch`` as the command to launch tasks.
    * Add support for pip install: ``pip3 install byteps``

## Performance

We show our experiment on BERT-large training, which is based on GluonNLP toolkit. The model uses mixed precision.

We use Tesla V100 32GB GPUs and set batch size equal to 64 per GPU. Each machine has 8 V100 GPUs (32GB memory) with NVLink-enabled. Machines are inter-connected with 100 Gbps RDMA network. This is the same hardware setup you can get on [AWS](https://aws.amazon.com/about-aws/whats-new/2018/12/introducing-amazon-ec2-p3dn-instances-our-most-powerful-gpu-instance-yet/).

BytePS achieves ~90% scaling efficiency for BERT-large with 256 GPUs. The code is available [here](https://github.com/ymjiang/gluon-nlp/tree/bert-byteps/scripts/bert). As a comparison, Horovod+NCCL has only ~70% scaling efficiency even after expert parameter tunning.

![BERT-Large](https://user-images.githubusercontent.com/13852819/69874496-1ca43600-12f6-11ea-997b-b023e4c93360.png)


With slower network, BytePS offers even more performance advantages -- up to 2x of Horovod+NCCL. You can find more evaluation results at [performance.md](docs/performance.md).

## Goodbye MPI, Hello Cloud

How can BytePS outperform Horovod by so much? One of the main reasons is that BytePS is designed for cloud and shared clusters, and throws away MPI.

MPI was born in the HPC world and is good for a cluster built with homogeneous hardware and for running a single job. However, cloud (or in-house shared clusters) is different.

This leads us to rethink the best communication strategy, as explained in [here](docs/rationale.md). In short, BytePS only uses NCCL inside a machine, while re-implements the inter-machine communication.

BytePS also incorporates many acceleration techniques such as hierarchical strategy, pipelining, tensor partitioning, NUMA-aware local communication, priority-based scheduling, etc.

## Quick Start

We provide a [step-by-step tutorial](docs/step-by-step-tutorial.md) for you to run benchmark training tasks. The simplest way to start is to use our [docker images](docker). Refer to [Documentations](docs) for how to [launch distributed jobs](docs/running.md) and more [detailed configurations](docs/env.md). After you can start BytePS, read [best practice](docs/best-practice.md) to get the best performance.

Below, we explain how to install BytePS by yourself. There are two options.

### Install by pip

```
pip3 install byteps
```

### Build from source code

You can try out the latest features by directly installing from master branch:

```
git clone --recursive https://github.com/bytedance/byteps
cd byteps
python3 setup.py install
```

Notes for above two options:
- BytePS assumes that you have already installed one or more of the following frameworks: TensorFlow / PyTorch / MXNet.
- BytePS depends on CUDA and NCCL. You should specify the NCCL path with `export BYTEPS_NCCL_HOME=/path/to/nccl`. By default it points to `/usr/local/nccl`.
- The installation requires gcc>=4.9. If you are working on CentOS/Redhat and have gcc<4.9, you can try `yum install devtoolset-7` before everything else. In general, we recommend using gcc 4.9 for best compatibility ([how to pin gcc](https://github.com/bytedance/byteps/blob/3fba75def0d81c1d3225f8f397cc985200f57de7/docker/Dockerfile.mxnet#L72-L80)).
- RDMA support: During setup, the script will automatically detect the RDMA header file. If you want to use RDMA, make sure your RDMA environment has been properly installed and tested before install ([install on Ubuntu-18.04](https://github.com/bytedance/byteps/blob/3fba75def0d81c1d3225f8f397cc985200f57de7/docker/Dockerfile.mxnet#L29-L33)).

## Examples

Basic examples are provided under the [example](example) folder. 

To reproduce the end-to-end evaluation in our OSDI'20 paper, find the code at this [repo](https://github.com/byteps/examples).

## Use BytePS in Your Code

Though being totally different at its core, BytePS is highly compatible with Horovod interfaces (Thank you, Horovod community!). We chose Horovod interfaces in order to minimize your efforts for testing BytePS.

If your tasks only rely on Horovod's allreduce and broadcast, you should be able to switch to BytePS in 1 minute. Simply replace `import horovod.tensorflow as hvd` by `import byteps.tensorflow as bps`, and then replace all `hvd` in your code by `bps`. If your code invokes `hvd.allreduce` directly, you should also replace it by `bps.push_pull`.

Many of our examples were copied from Horovod and modified in this way. For instance, compare the MNIST example for [BytePS](https://github.com/bytedance/byteps/blob/master/example/tensorflow/tensorflow_mnist.py) and [Horovod](https://github.com/horovod/horovod/blob/master/examples/tensorflow_mnist.py).

BytePS also supports other native APIs, e.g., PyTorch Distributed Data Parallel and TensorFlow Mirrored Strategy. See [DistributedDataParallel.md](docs/DistributedDataParallel.md) and [MirroredStrategy.md](docs/MirroredStrategy.md) for usage.

## Limitations and Future Plans
BytePS does not support pure CPU training for now. One reason is that the [cheap PS assumption](docs/rationale.md) of BytePS do not hold for CPU training. Consequently, you need CUDA and NCCL to build and run BytePS.

We would like to have below features, and there is no fundamental difficulty to implement them in BytePS architecture. However, they are not implemented yet:
* Sparse model training
* Fault-tolerance
* Straggler-mitigation

## Publications

1. [OSDI'20] "[A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters](https://www.usenix.org/conference/osdi20/presentation/jiang)". Yimin Jiang, Yibo Zhu, Chang Lan, Bairen Yi, Yong Cui, Chuanxiong Guo. (Paper to appear soon)

2. [SOSP'19] "[A Generic Communication Scheduler for Distributed DNN Training Acceleration](https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf)". Yanghua Peng, Yibo Zhu, Yangrui Chen, Yixin Bao, Bairen Yi, Chang Lan, Chuan Wu, Chuanxiong Guo. (Code is at [bytescheduler branch](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler))
