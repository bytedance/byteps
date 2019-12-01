# BytePS

[![Build Status](https://travis-ci.org/bytedance/byteps.svg?branch=master)](https://travis-ci.org/bytedance/byteps)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

BytePS is a high performance and general distributed training framework. It supports TensorFlow, Keras, PyTorch, and MXNet, and can run on either TCP or RDMA network.

BytePS outperforms existing open-sourced distributed training frameworks by a large margin. For example, on BERT-large training, BytePS can achieve ~90% scaling efficiency with 256 GPUs (see below), which is much higher than [Horovod](https://github.com/horovod/horovod)+[NCCL](https://github.com/NVIDIA/nccl).

## News

- [New Server](https://github.com/bytedance/byteps/pull/151): We improve the server performance by a large margin, and it is now independent of MXNet KVStore. Try our [new docker images](docker/).
- Use [the ssh launcher](launcher/) to launch your distributed jobs
- [Improved key distribution strategy for better load-balancing](https://github.com/bytedance/byteps/pull/116)
- [Improved RDMA robustness](https://github.com/bytedance/byteps/pull/91)

## Performance

We show our experiment on BERT-large training, which is based on GluonNLP toolkit. The model uses mixed precision.

We use Tesla V100 32GB GPUs and set batch size equal to 64 per GPU. Each machine has 8 V100 GPUs (32GB memory) with NVLink-enabled. Machines are inter-connected with 100 Gbps RoCEv2 network.

BytePS achieves ~90% scaling efficiency for BERT-large. The code is available [here](https://github.com/ymjiang/gluon-nlp/tree/bert-byteps/scripts/bert). 

![BERT-Large](https://user-images.githubusercontent.com/13852819/69874496-1ca43600-12f6-11ea-997b-b023e4c93360.png)


More evaluation in different scenarios can be found at [performance.md](docs/performance.md).

## Goodbye MPI, Hello Cloud

How can BytePS outperform Horovod by so much? One of the main reasons is that BytePS is designed for cloud and shared clusters, and throws away MPI.

MPI was born in the HPC world and is good for a cluster built with homogeneous hardware and for running a single job. However, cloud (or in-house shared clusters) is different.

This leads us to rethink the best communication strategy, as explained in [here](docs/rationale.md). In short, BytePS only uses NCCL inside a machine, while re-implements the inter-machine communication.

BytePS also incorporates many acceleration techniques such as hierarchical strategy, pipelining, tensor partitioning, NUMA-aware local communication, priority-based scheduling, etc.

## Quick Start

We provide a [step-by-step tutorial](docs/step-by-step-tutorial.md) for you to run benchmark training tasks. After you can start BytePS, read [best practice](docs/best-practice.md) to get the best performance.

Below, we explain how to build and run BytePS by yourself. BytePS assumes that you have already installed one or more of the following frameworks: TensorFlow / PyTorch / MXNet. BytePS depends on CUDA and NCCL, and requires gcc>=4.9. If you are working on CentOS/Redhat and have gcc<4.9, you can try `yum install devtoolset-7` before everything else.


### Build from wheel 

You can download our wheels and install. Please refer to [pip-list.md](docs/pip-list.md) for more instructions.

### Build from source code

If the above does not contain your desired wheel resource, or you want to try building from source code: 

```
git clone --recurse-submodules https://github.com/bytedance/byteps
cd byteps
python setup.py install
```

Notes:
- Please pin your gcc to 4.9 before building, [here](https://github.com/bytedance/byteps/blob/master/docker/Dockerfile.worker.pytorch.cu100#L123-L131) is an example.
- You may set `BYTEPS_USE_RDMA=1` to install with RDMA support. Before this, make sure your RDMA drivers have been properly installed and tested.


For your server and scheduler node, we highly recommend you to just use our prebuilt docker image `bytepsimage/byteps_server` (TCP) or `bytepsimage/byteps_server_rdma` (RDMA). Otherwise, you have to manually compile our modified [MXNet](https://github.com/bytedance/incubator-mxnet) as in our dockerfiles: [Dockerfile.server](docker/Dockerfile.server) and [Dockerfile.server.rdma](docker/Dockerfile.server.rdma). 

Refer to [Documentations](docs) for how to [launch distributed jobs](docs/running.md) and more [detailed configurations](docs/env.md).

## Use BytePS in Your Code

Though being totally different at its core, BytePS is highly compatible with Horovod interfaces (Thank you, Horovod community!). We chose Horovod interfaces in order to minimize your efforts for testing BytePS.

If your tasks only rely on Horovod's allreduce and broadcast, you should be able to switch to BytePS in 1 minute. Simply replace `import horovod.tensorflow as hvd` by `import byteps.tensorflow as bps`, and then replace all `hvd` in your code by `bps`. If your code invokes `hvd.allreduce` directly, you should also replace it by `bps.push_pull`.

Many of our examples were copied from Horovod and modified in this way. For instance, compare the MNIST example for [BytePS](https://github.com/bytedance/byteps/blob/master/example/tensorflow/tensorflow_mnist.py) and [Horovod](https://github.com/horovod/horovod/blob/master/examples/tensorflow_mnist.py).

## Limitations and Future Plans
BytePS does not support pure CPU training for now. One reason is that the [cheap PS assumption](docs/rationale.md) of BytePS do not hold for CPU training. Consequently, you need CUDA and NCCL to build and run BytePS.

We would like to have below features, and it is not hard to implement them in BytePS architecture. However, they are not implemented yet:
* Sparse model training
* Fault-tolerance
* Straggler-mitigation

## Publications
BytePS adopts similar ideas in [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler), e.g., tensor partitioning and credit-based preemptive scheduling, but with a different system design as it works as a communication library under the framework engine layer. To access ByteScheduler's source code, check the bytescheduler folder in bytescheduler branch of this repo [here](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler). You can also find more details about ByteScheduler in the following [paper](https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf):

Yanghua Peng, Yibo Zhu, Yangrui Chen, Yixin Bao, Bairen Yi, Chang Lan, Chuan Wu, Chuanxiong Guo. "A Generic Communication Scheduler for Distributed DNN Training Acceleration," in ACM SOSP, Huntsville, Ontario, Canada, October 27-30, 2019.
