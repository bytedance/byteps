# A Step-by-Step Tutorial

The goal of this tutorial is to help you run BytePS quickly. To ensure that you don't get trouble with system environments problem, we recommend you to use our provided images.

When you have successfully run through these examples, read the [best practice](./best-practice.md) and [performance tuning tips](https://github.com/bytedance/byteps/blob/master/docs/env.md#performance-tuning) to get the best performance on your setup.

## Single Machine Training

### TensorFlow
```
docker pull bytepsimage/tensorflow

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/tensorflow bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # gpus list
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # one worker
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1
export DMLC_PS_ROOT_PORT=1234

bpslaunch python3 /usr/local/byteps/example/tensorflow/synthetic_benchmark.py --model ResNet50 --num-iters 1000000
```

### PyTorch


```
docker pull bytepsimage/pytorch

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/pytorch bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # gpus list
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # one worker
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1
export DMLC_PS_ROOT_PORT=1234

bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py --model resnet50 --num-iters 1000000
```

### MXNet

```
docker pull bytepsimage/mxnet

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/mxnet bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # gpus list
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # one worker
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1
export DMLC_PS_ROOT_PORT=1234

bpslaunch python3 /usr/local/byteps/example/mxnet/train_imagenet_byteps.py --benchmark 1 --batch-size=32
```

## Distributed Training (TCP)

Let's say you have two workers, and each one with 4 GPUs. For simplicity we use one server. In practice, you need more servers (at least equal to the number of workers) to achieve high performance.


For the workers, you need to pay attention to `DMLC_WORKER_ID`. This is the main difference compared to single machine jobs. Let's say the 2 workers are using TensorFlow.

For the scheduler:
```
docker pull bytepsimage/tensorflow

docker run -it --net=host bytepsimage/tensorflow bash

# now you are in docker environment
export DMLC_NUM_WORKER=2
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234  # the scheduler port

bpslaunch
```

For the server:
```
docker pull bytepsimage/tensorflow

docker run -it --net=host bytepsimage/tensorflow bash

# now you are in docker environment
export DMLC_NUM_WORKER=2
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234  # the scheduler port

bpslaunch
```


For worker-0:
```
docker pull bytepsimage/tensorflow

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/tensorflow bash

export NVIDIA_VISIBLE_DEVICES=0,1,2,3
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234 # the scheduler port

bpslaunch python3 /usr/local/byteps/example/tensorflow/synthetic_benchmark.py --model ResNet50 --num-iters 1000000
```

For worker-1:

```
docker pull bytepsimage/tensorflow

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/tensorflow bash

export NVIDIA_VISIBLE_DEVICES=0,1,2,3
export DMLC_WORKER_ID=1
export DMLC_NUM_WORKER=2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234 # the scheduler port

bpslaunch python3 /usr/local/byteps/example/tensorflow/synthetic_benchmark.py --model ResNet50 --num-iters 1000000
```


If your workers use PyTorch, you need to change the image name to `bytepsimage/pytorch`, and replace the python script of the workers with

```
bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py --model resnet50 --num-iters 1000000
```


If your workers use MXNet, you need to change the image name to `bytepsimage/mxnet`, and replace the python script of the workers with
```
bpslaunch python3 /usr/local/byteps/example/mxnet/train_imagenet_byteps.py --benchmark 1 --batch-size=32
```

## Distributed Training with RDMA

The steps to launch RDMA tasks are basically similar to the above. The main differences are that (1) you need to specify your RDMA devices when running a docker, and (2) you need to set `DMLC_ENABLE_RDMA=1`.

In the following, let's continue to use the example: you have two workers and one server, and the workers are using TensorFlow.

For the scheduler:
```
docker pull bytepsimage/tensorflow

# specify your rdma device (usually under /dev/infiniband, but depend on your system configurations)
docker run -it --net=host --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/tensorflow bash

# now you are in docker environment
export DMLC_ENABLE_RDMA=1
export DMLC_NUM_WORKER=2
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=1

# the RDMA interface name of the scheduler
export DMLC_INTERFACE=eth5

# your scheduler's RDMA NIC information (IP, port)
export DMLC_PS_ROOT_URI=10.0.0.100
export DMLC_PS_ROOT_PORT=9000

# launch the job
bpslaunch
```

For the server:
```
docker pull bytepsimage/tensorflow

# specify your rdma device (usually under /dev/infiniband, but depend on your system configurations)
docker run -it --net=host --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/tensorflow bash

# now you are in docker environment
export DMLC_ENABLE_RDMA=1
export DMLC_NUM_WORKER=2
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1

# the RDMA interface name of the server
export DMLC_INTERFACE=eth5

# your scheduler's RDMA NIC information (IP, port)
export DMLC_PS_ROOT_URI=10.0.0.100
export DMLC_PS_ROOT_PORT=9000

# launch the job
bpslaunch
```

For worker-0:

```
docker pull bytepsimage/tensorflow

# specify your rdma device (usually under /dev/infiniband, but depend on your system configurations)
nvidia-docker run -it --net=host --shm-size=32768m --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/tensorflow bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3

export DMLC_ENABLE_RDMA=1
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1

# the RDMA interface name of this worker
export DMLC_INTERFACE=eth5

# your scheduler's RDMA NIC information (IP, port)
export DMLC_PS_ROOT_URI=10.0.0.100
export DMLC_PS_ROOT_PORT=9000

# launch the job
bpslaunch python3 /usr/local/byteps/example/tensorflow/synthetic_benchmark.py --model ResNet50 --num-iters 1000000
```

For worker-1:


```
docker pull bytepsimage/tensorflow

# specify your rdma device (usually under /dev/infiniband, but depend on your system configurations)
nvidia-docker run -it --net=host --shm-size=32768m --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/tensorflow bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3

export DMLC_ENABLE_RDMA=1
export DMLC_WORKER_ID=1
export DMLC_NUM_WORKER=2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1

# the RDMA interface name of this worker
export DMLC_INTERFACE=eth5

# your scheduler's RDMA NIC information (IP, port)
export DMLC_PS_ROOT_URI=10.0.0.100
export DMLC_PS_ROOT_PORT=9000

# launch the job
bpslaunch python3 /usr/local/byteps/example/tensorflow/synthetic_benchmark.py --model ResNet50 --num-iters 1000000
```



If your workers use PyTorch, you need to change the image name to `bytepsimage/pytorch`, and replace the python script of the workers with

```
bpslaunch python3 /usr/local/byteps/example/pytorch/benchmark_byteps.py --model resnet50 --num-iters 1000000
```


If your workers use MXNet, you need to change the image name to `bytepsimage/mxnet`, and replace the python script of the workers with
```
bpslaunch python3 /usr/local/byteps/example/mxnet/train_imagenet_byteps.py --benchmark 1 --batch-size=32
```