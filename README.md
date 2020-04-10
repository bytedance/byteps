# BytePS

[![Build Status](https://travis-ci.org/bytedance/byteps.svg?branch=master)](https://travis-ci.org/bytedance/byteps)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Pypi](https://img.shields.io/pypi/v/byteps.svg)

BytePS is a high performance and general distributed training framework. It supports TensorFlow, Keras, PyTorch, and MXNet, and can run on either TCP or RDMA network.

BytePS outperforms existing open-sourced distributed training frameworks by a large margin. For example, on BERT-large training, BytePS can achieve ~90% scaling efficiency with 256 GPUs (see below), which is much higher than [Horovod](https://github.com/horovod/horovod)+[NCCL](https://github.com/NVIDIA/nccl). In certain scenarios, BytePS can double the training speed compared with Horovod+NCCL.

## News

- [BytePS-0.2.0](CHANGELOG.rst) has been released.
- Now pip install is available, refer to the [install tutorial](https://github.com/bytedance/byteps#quick-start).
- [Largely improve RDMA performance](https://github.com/bytedance/byteps/pull/184). Now support colocating servers and workers with high performance.
- Fix [RDMA fork problem](https://github.com/bytedance/byteps/pull/192) caused by multi-processing.
- [New Server](https://github.com/bytedance/byteps/pull/151): We improve the server performance by a large margin, and it is now independent of MXNet KVStore. Try our [new docker images](docker/).
- Use [the ssh launcher](launcher/) to launch your distributed jobs
- [Improved key distribution strategy for better load-balancing](https://github.com/bytedance/byteps/pull/116)
- [Improved RDMA robustness](https://github.com/bytedance/byteps/pull/91)

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
- The installation requires gcc>=4.9. If you are working on CentOS/Redhat and have gcc<4.9, you can try `yum install devtoolset-7` before everything else. In general, we recommend using gcc 4.9 for best compatibility ([an example](https://github.com/bytedance/byteps/blob/3fba75def0d81c1d3225f8f397cc985200f57de7/docker/Dockerfile.mxnet#L72-L80) to pin gcc).
- RDMA support: During setup, the script will automatically detect the RDMA header file. If you want to use RDMA, make sure your RDMA environment has been properly installed and tested before install ([an example](https://github.com/bytedance/byteps/blob/3fba75def0d81c1d3225f8f397cc985200f57de7/docker/Dockerfile.mxnet#L29-L33) for Ubuntu-18.04).


## Use BytePS in Your Code

Though being totally different at its core, BytePS is highly compatible with Horovod interfaces (Thank you, Horovod community!). We chose Horovod interfaces in order to minimize your efforts for testing BytePS.

If your tasks only rely on Horovod's allreduce and broadcast, you should be able to switch to BytePS in 1 minute. Simply replace `import horovod.tensorflow as hvd` by `import byteps.tensorflow as bps`, and then replace all `hvd` in your code by `bps`. If your code invokes `hvd.allreduce` directly, you should also replace it by `bps.push_pull`.

Many of our examples were copied from Horovod and modified in this way. For instance, compare the MNIST example for [BytePS](https://github.com/bytedance/byteps/blob/master/example/tensorflow/tensorflow_mnist.py) and [Horovod](https://github.com/horovod/horovod/blob/master/examples/tensorflow_mnist.py).

## Limitations and Future Plans
BytePS does not support pure CPU training for now. One reason is that the [cheap PS assumption](docs/rationale.md) of BytePS do not hold for CPU training. Consequently, you need CUDA and NCCL to build and run BytePS.

We would like to have below features, and there is no fundamental difficulty to implement them in BytePS architecture. However, they are not implemented yet:
* Sparse model training
* Fault-tolerance
* Straggler-mitigation

## Publications
BytePS adopts similar ideas in [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler), e.g., tensor partitioning and credit-based preemptive scheduling, but with a different system design as it works as a communication library under the framework engine layer. To access ByteScheduler's source code, check the bytescheduler folder in bytescheduler branch of this repo [here](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler). You can also find more details about ByteScheduler in the following [paper](https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf):

Yanghua Peng, Yibo Zhu, Yangrui Chen, Yixin Bao, Bairen Yi, Chang Lan, Chuan Wu, Chuanxiong Guo. "A Generic Communication Scheduler for Distributed DNN Training Acceleration," in ACM SOSP, Huntsville, Ontario, Canada, October 27-30, 2019.
