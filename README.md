# BytePS

BytePS is a high performance and general distributed training framework that supports TensorFlow, Keras, PyTorch, and MXNet. 
BytePS achieves higher performance than existing Parameter Server (PS, e.g., MXNet-KVstore) or Collective-based frameworks (e.g., Horovod). 
BytePS supports both TCP and RDMA. 


BytePS achieves high performance by leveraging a hybrid communication strategy: combines PS and Collective primitives together, and incorporates acceleration techniques such as pipelining, tensor partitioning, NUMA-aware local communication and priority-based scheduling, etc. 

* [Documentations](docs)
* [Dockerfiles & Installations](docker)
* [Examples](example)

# Quick Start

Before using BytePS, we assume you have already installed one or many of the following frameworks: TensorFlow / Keras / PyTorch / MXNet.
 
Clone BytePS and its third party dependency:

```
git clone --recurse-submodules https://github.com/bytedance/byteps
```

Then `cd` into your BytePS directory and install. 
You may set `BYTEPS_USE_RDMA=1` to install with RDMA support. 
```
python setup.py install
```

Now you can try our examples. Let's say you are using MXNet and want to try a Resnet50 training benchmark:

```
export NVIDIA_VISIBLE_DEVICES=0,1 \
       DMLC_NUM_WORKER=1 \
       DMLC_NUM_SERVER=1 \
       DMLC_WORKER_ID=0 \
       DMLC_ROLE=worker \
       DMLC_PS_ROOT_URI=10.0.0.1 \
       DMLC_PS_ROOT_PORT=1234 \
       DMLC_INTERFACE=eth0 
       
python byteps/launcher/launch.py byteps/example/mxnet/train_imagenet_horovod.py --benchmark 1 --batch-size=32 
```

Refer to [Documentations](docs) for more hands-on tutorials.


# Performance

We choose two models for performance evaluation: VGG16 (communication-intensive) and Resnet50 (computation-intensive). 
We use Tesla V100 GPUs and set each GPU with batch size equals 64. The ideal training speed is calculated by: `# GPUs * single GPU training speed`.

Below shows the performance on NVLink-enabled machines (each machine has 8 GPUs). Machines are inter-connected with 20 Gbit/s TCP networking. 
In this case, the network bandwidth is the bottleneck. BytePS outperforms Horovod by 44% for Resnet50, and 100% for VGG16. 
This is because BytePS uses additional computation resources (PS) to aggregate and reduce traffic.

<img src="/images/perf_tcp_vgg16.png" width="360" height="220"><img src="/images/perf_tcp_resnet50.png" width="360" height="220">


We further evaluate BytePS on RDMA-enabled networks. The machines are based on PCIe architecture (4 GPUs under one PCIe switch), and each machine contains two PCIe switches.
The machines are inter-connected by 100 Gbps RoCEv2 networks.
The bottleneck is the local PCIe bandwidth.
In this case, BytePS outperforms Horovod by 7% for Resnet50, and 17% for VGG16. 
The performance gain is due to BytePS's NUMA-aware techniques that reduces cross-PCIe switch traffic.  

<img src="/images/perf_rdma_vgg16.png" width="360" height="220"><img src="/images/perf_rdma_resnet50.png" width="360" height="220">

