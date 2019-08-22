# A Step-by-Step Tutorial 

The goal of this tutorial is to help you run BytePS quickly. To ensure that you don't get trouble with system environments problem, we recommend you to use our provided images.

When you have successfully run through these examples, read the [best practice](./best-practice.md) and [performance tuning tips](https://github.com/bytedance/byteps/blob/master/docs/env.md#performance-tuning) to get the best performance on your setup.
 
## Single Machine Training 

### TensorFlow
```
docker pull bytepsimage/worker_tensorflow

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/worker_tensorflow bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # say you have 4 GPUs 
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # you only have one worker
export DMLC_ROLE=worker # your role is worker

# the following value does not matter for non-distributed jobs 
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 
export DMLC_PS_ROOT_PORT=1234 

# can also try: export EVAL_TYPE=mnist 
export EVAL_TYPE=benchmark 
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/tensorflow/run_tensorflow_byteps.sh \
       --model ResNet50 --num-iters 1000000        
```

### PyTorch


```
docker pull bytepsimage/worker_pytorch

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/worker_pytorch bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # say you have 4 GPUs 
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # you only have one worker
export DMLC_ROLE=worker # your role is worker

# the following value does not matter for non-distributed jobs 
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 
export DMLC_PS_ROOT_PORT=1234 

export EVAL_TYPE=benchmark 
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/pytorch/start_pytorch_byteps.sh \
       --model resnet50 --num-iters 1000000      
```

### MXNet

```
docker pull bytepsimage/worker_mxnet

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/worker_mxnet bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # say you have 4 GPUs 
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # you only have one worker
export DMLC_ROLE=worker # your role is worker

# the following value does not matter for non-distributed jobs 
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 
export DMLC_PS_ROOT_PORT=1234 

export EVAL_TYPE=benchmark 
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/mxnet/start_mxnet_byteps.sh \
       --benchmark 1 --batch-size=32  
```

## Distributed Training (TCP)

Let's say you have two workers, and each one with 4 GPUs. For simplicity we use one server.

The way to launch the scheduler and the server are the same for any framework.

For the scheduler:
```
# scheduler can use the same image as servers
docker pull bytepsimage/byteps_server

docker run -it --net=host bytepsimage/byteps_server bash

# now you are in docker environment
export DMLC_NUM_WORKER=2 
export DMLC_ROLE=scheduler 
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP 
export DMLC_PS_ROOT_PORT=1234  # the scheduler port

python /usr/local/byteps/launcher/launch.py
```

For the server:
```
docker pull bytepsimage/byteps_server

docker run -it --net=host bytepsimage/byteps_server bash

# now you are in docker environment
export DMLC_NUM_WORKER=2 
export DMLC_ROLE=server  
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP 
export DMLC_PS_ROOT_PORT=1234  # the scheduler port

# 4 threads should be enough for a server
export MXNET_OMP_MAX_THREADS=4 

python /usr/local/byteps/launcher/launch.py
```

For the workers, you need to pay attention to `DMLC_WORKER_ID`. This is the main difference compared to single machine jobs. Let's say the 2 workers are using MXNet.

For worker-0:
```
docker pull bytepsimage/worker_mxnet

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/worker_mxnet bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # say you have 4 GPUs 
export DMLC_WORKER_ID=0 # worker-0
export DMLC_NUM_WORKER=2 # 2 workers
export DMLC_ROLE=worker # your role is worker
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP 
export DMLC_PS_ROOT_PORT=1234 # the scheduler port

export EVAL_TYPE=benchmark 
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/mxnet/start_mxnet_byteps.sh \
       --benchmark 1 --batch-size=32  
```

For worker-1:

```
docker pull bytepsimage/worker_mxnet

nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/worker_mxnet bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # say you have 4 GPUs 
export DMLC_WORKER_ID=1 # worker-1
export DMLC_NUM_WORKER=2 # 2 workers
export DMLC_ROLE=worker # your role is worker
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP 
export DMLC_PS_ROOT_PORT=1234 # the scheduler port

export EVAL_TYPE=benchmark 
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/mxnet/start_mxnet_byteps.sh \
       --benchmark 1 --batch-size=32  
```

If your workers use TensorFlow, you need to change the image name to `bytepsimage/worker_tensorflow`, and replace the python script with
```
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/tensorflow/run_tensorflow_byteps.sh \
       --model ResNet50 --num-iters 1000000     
```

If your workers use PyTorch, you need to change the image name to `bytepsimage/worker_pytorch`, and replace the python script with

```
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/pytorch/start_pytorch_byteps.sh \
       --model resnet50 --num-iters 1000000   
```

## Distributed Training with RDMA

The steps to launch RDMA tasks are basically similar to the above. The main differences are that (1) you need to specify your RDMA devices when running a docker, and (2) you need to set `DMLC_ENABLE_RDMA=1`. To run this example, your `nvidia-docker` need to support cuda 10.

In the following, let's continue to use the example: you have two workers and one server, and the workers are using MXNet. 

For the scheduler:
```
# the scheduler may use the same image as servers
docker pull bytepsimage/byteps_server_rdma

# specify your rdma device (usually under /dev/infiniband, but depend on your system configurations)
docker run -it --net=host --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/byteps_server_rdma bash

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
python /usr/local/byteps/launcher/launch.py
```

For the server:
```
docker pull bytepsimage/byteps_server_rdma

# specify your rdma device (usually under /dev/infiniband, but depend on your system configurations)
docker run -it --net=host --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/byteps_server_rdma bash

# now you are in docker environment
export DMLC_ENABLE_RDMA=1
export DMLC_NUM_WORKER=2
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1

# the RDMA interface name of the server
export DMLC_INTERFACE=eth5

# 4 threads should be enough for a server
export MXNET_OMP_MAX_THREADS=4 

# your scheduler's RDMA NIC information (IP, port)
export DMLC_PS_ROOT_URI=10.0.0.100
export DMLC_PS_ROOT_PORT=9000

# launch the job
python /usr/local/byteps/launcher/launch.py
```

For worker-0:

```
docker pull bytepsimage/worker_mxnet_rdma

# specify your rdma device (usually under /dev/infiniband, but depend on your system configurations)
nvidia-docker run -it --net=host --shm-size=32768m --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/worker_mxnet_rdma bash

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
export EVAL_TYPE=benchmark
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/mxnet/start_mxnet_byteps.sh \
       --benchmark 1 --batch-size=32  
```

For worker-1:


```
docker pull bytepsimage/worker_mxnet_rdma

# specify your rdma device (usually under /dev/infiniband, but depend on your system configurations)
nvidia-docker run -it --net=host --shm-size=32768m --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/worker_mxnet_rdma bash

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
export EVAL_TYPE=benchmark
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/mxnet/start_mxnet_byteps.sh \
       --benchmark 1 --batch-size=32  
```


If your workers use TensorFlow, you need to change the image name to `bytepsimage/worker_tensorflow_rdma`, and replace the python script with
```
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/tensorflow/run_tensorflow_byteps.sh \
       --model ResNet50 --num-iters 1000000     
```

If your workers use PyTorch, you need to change the image name to `bytepsimage/worker_pytorch_rdma`, and replace the python script with

```
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/pytorch/start_pytorch_byteps.sh \
       --model resnet50 --num-iters 1000000   
```