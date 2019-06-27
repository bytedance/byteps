# A Step-by-Step Tutorial 

The goal of this tutorial is to help you run BytePS quickly. To ensure that you don't get trouble with system environments problem, we recommend you to use our provided images (as the first step).

 
## Single Machine Training 

### TensorFlow
```
docker pull bytepsimage/worker_tensorflow

nvidia-docker run --shm-size=32768m -it bytepsimage/worker_tensorflow bash

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
       --model ResNet50 --num-iters 1000        
```

### PyTorch


```
docker pull bytepsimage/worker_pytorch

nvidia-docker run --shm-size=32768m -it bytepsimage/worker_pytorch bash

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
       --model resnet50 --num-iters 1000      
```

### MXNet

```
docker pull bytepsimage/worker_mxnet

nvidia-docker run --shm-size=32768m -it bytepsimage/worker_mxnet bash

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

## Distributed Training 

Let's say you have two workers, and each one with 4 GPUs. For simplicity we use one server.

The way to launch the scheduler and the server are the same for any framework.

For the scheduler:
```
# scheduler can use the same image as servers
docker pull bytepsimage/byteps_server

docker run -it bytepsimage/byteps_server bash

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

docker run -it bytepsimage/byteps_server bash

# now you are in docker environment
export DMLC_NUM_WORKER=2 
export DMLC_ROLE=server  
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 # the scheduler IP 
export DMLC_PS_ROOT_PORT=1234  # the scheduler port

python /usr/local/byteps/launcher/launch.py
```

For the workers, you need to pay attention to `DMLC_WORKER_ID`. This is the main difference compared to single machine jobs. Let's say the 2 workers are using MXNet.

For worker-0:
```
docker pull bytepsimage/worker_mxnet

nvidia-docker run --shm-size=32768m -it bytepsimage/worker_mxnet bash

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

nvidia-docker run --shm-size=32768m -it bytepsimage/worker_mxnet bash

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
       --model ResNet50 --num-iters 1000     
```

If your workers use PyTorch, you need to change the image name to `bytepsimage/worker_pytorch`, and replace the python script with

```
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/pytorch/start_pytorch_byteps.sh \
       --model resnet50 --num-iters 1000   
```

