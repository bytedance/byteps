# BytePS

BytePS is a high performance and general distributed training framework that supports TensorFlow, Keras, PyTorch, and MXNet. 
BytePS achieves higher performance than existing Parameter Server (PS, e.g., MXNet-KVstore) or Collective-based frameworks (e.g., Horovod). 
BytePS supports both TCP and RDMA. 


BytePS achieves high performance by leveraging a hybrid communication strategy: combines PS and Collective primitives together, 
and incorporates acceleration techniques such as pipelining, tensor partitioning, NUMA-aware local communication and priority-based scheduling, etc. 

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

Now you can try our [examples](example). Let's say you are using MXNet and want to try a Resnet50 training benchmark:

```
export NVIDIA_VISIBLE_DEVICES=0,1 \
       DMLC_NUM_WORKER=1 \
       DMLC_NUM_SERVER=1 \
       DMLC_WORKER_ID=0 \
       DMLC_ROLE=worker \
       DMLC_PS_ROOT_URI=10.0.0.1 \
       DMLC_PS_ROOT_PORT=1234 \
       DMLC_INTERFACE=eth0 
       
python byteps/launcher/launch.py byteps/example/mxnet/train_imagenet_byteps.py --benchmark 1 --batch-size=32 
```

For distributed training, you also need to build a server image. We provide [Dockerfiles](docker) as examples. 
You may use the same images for the scheduler and the servers.

Refer to [Documentations](docs) for how to launch distributed jobs and more hands-on tutorials.

# Performance

We choose two models for performance evaluation: VGG16 (communication-intensive) and Resnet50 (computation-intensive). 
We use Tesla V100 GPUs and set each GPU with batch size equals 64. 

Below shows the performance on NVLink-enabled machines (each machine has 8 GPUs). Machines are inter-connected with 20 Gbit/s TCP networking.
BytePS outperforms Horovod (NCCL) by 44% for Resnet50, and 100% for VGG16. 

<img src="/docs/images/perf_tcp_vgg16.png" width="360" height="220"><img src="/docs/images/perf_tcp_resnet50.png" width="360" height="220">

Evaluation on RDMA networks can be found at [performance.md](docs).   