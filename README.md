# BytePS

BytePS is a high performance and general distributed training framework that supports TensorFlow, Keras, PyTorch, and MXNet. 
BytePS achieves higher performance than existing Parameter Server (PS) or Collective-based frameworks (such as MXNet-KVstore and Horovod). BytePS supports both TCP and RDMA. 


BytePS achieves high performance because we leverage a hybrid communication strategy: combines PS and Collective primitives together, and incorporates acceleration techniques such as pipelining, tensor partitioning, NUMA-aware local communication and priority-based scheduling, etc. 

* [Documentations](docs)
* [Dockerfiles & Installations](docker)
* [Examples](example)

# Quick Start

Before using BytePS, we assume you have already installed one or many of the following frameworks: TensorFlow / Keras / PyTorch / MXNet.
 
First you need to clone and install [ps-lite](https://github.com/bytedance/ps-lite) as third party dependency:

```
git clone --recurse-submodules https://github.com/bytedance/byteps
cd byteps/3rdparty/ps-lite && make -j4
```

Then install BytePS:

```
python byteps/setup.py install
```

Now you can try our examples. Let's say you are using MXNet and want to try a Resnet50 training benchmark:

```
export NVIDIA_VISIBLE_DEVICES=0,1 DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1 DMLC_WORKER_ID=0 DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.0.0.1 DMLC_PS_ROOT_PORT=1234 DMLC_INTERFACE=eth0 
python byteps/launcher/launch.py  byteps/example/mxnet/train_imagenet_horovod.py --benchmark 1 --batch-size=32 
```

# Performance

Below shows BytePS's performance on a PCIe machine (8 GPUs). BytePS beats Horovod by 20% on VGG16 (communication-intensive) and 10% on Resnet50 (computation-intensive). 

<img src="/images/perf_8gpu.png" width="360" height="240">


For distributed training with RDMA (two workers, each with 8 GPUs), BytePS outperforms Horovod by 10-50%.

For distributed training with TCP (two workers, each with 8 GPUs), since the networking bandwidth is the bottleneck, the performance gain of BytePS is even higher. For example, BytePS outperforms Horovod by 100% on Alexnet training.

<img src="/images/perf_16gpu_rdma.png" width="360" height="240"><img src="/images/perf_16gpu_tcp.png" width="360" height="240">





