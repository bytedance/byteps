# BytePS Environment Variables

Regardless of your framework, TensorFlow, PyTorch or MXNet, you must set the required envrionment variables below, including DMLC_* variables. This is because we leverage the [DMLC/MXNet bootstrapping process](https://mxnet.apache.org/api/faq/distributed_training#manually-launching-jobs).

To run distributed training, you must start one scheduler, at least one server, and at least two workers. If you only have one worker, you won't need scheduler or server.

## Required for workers

For each worker machine, you must specify the following variables:

```
export DMLC_ROLE=worker
export DMLC_PS_ROOT_URI=a.b.c.d
export DMLC_PS_ROOT_PORT=p
export DMLC_WORKER_ID=x
export DMLC_NUM_WORKER=y
```

`DMLC_PS_ROOT_URI` is the IP of your scheduler. `DMLC_PS_ROOT_PORT` is the port that your scheduler binds to.

If you have `NVIDIA_VISIBLE_DEVICES` set, you can run `launcher/launcher.py YOUR_COMMAND` to start your job.

Alternatively, if you don't use `launcher/launcher.py`, you can start the job on each GPU after specifying:

```
export BYTEPS_LOCAL_RANK=r
export BYTEPS_LOCAL_SIZE=s
```

If you have RDMA network available, you should set:

```
DMLC_ENABLE_RDMA=1
```

Otherwise, set it to 0.

## Required for servers and scheduler

BytePS uses the same environment variables as MXNet for server and scheduler:
https://mxnet.apache.org/api/faq/distributed_training#manually-launching-jobs

In short, you should configure the same DMLC_* variables as the worker, except that DMLC_ROLE should be either server or scheduler.

Also, set DMLC_ENABLE_RDMA if you have RDMA network. This must be consistent with workers. Note that MXNet in the server and scheduler must be built with `USE_RDMA=1`. Please check out this [Dockerfile](https://github.com/bytedance/byteps/blob/master/docker/Dockerfile.server#L27) as an example.

## BytePS debug

If you are using launcher.py, you can enable gdb and get the backtrace (if the program terminates abnormally) by setting:

```
export BYTEPS_ENABLE_GDB=1
```

You can let BytePS print verbose logs by setting:

```
export BYTEPS_LOG_LEVEL=INFO
```

You can also let BytePS print values of a given tensor (specified by a key in integer) during different stages and iterations:

```
export BYTEPS_DEBUG_SAMPLE_TENSOR=xxxx
```

By default, if there is only one worker machine, BytePS won't connect to servers or schedulers because it is not needed. However, for debug purposes, you can force the worker to push and pull:

```
export BYTEPS_FORCE_DISTRIBUTED=1
```

The logging in the ps-lite middleware and on the server side is controlled by PS_VERBOSE. You can set the following to enable verbose output:

```
export PS_VERBOSE=2
```

## Performance tuning

There are several knobs that may impact the performance of BytePS. If you are not sure what they mean, you can leave them unmodified, i.e., by not setting them.

The most important one is the number of GPUs per PCIe switches. You should configure it according to your hardware. However, if you really do not know the hardware setup, you can leave it unmodified. BytePS should work as well as Horovod in that case, although a correct configuration may give you better performance than Horovod.

```
export BYTEPS_PCIE_SWITCH_SIZE=x
```

You can also configure the tensor partition size. A smaller size improves BytePS pipelining, but may have higher other overhead like NCCL coordination, ZMQ message headers, etc. The default and recommended value is 4096000 (in bytes).

```
export BYTEPS_PARTITION_BYTES=y
```

The rest do not impact the performance much. However, you can still experiment them if you have time.

You can increase the number of concurrent NCCL streams used in local merging. However, this may lead to occasional hanging problem due to NCCL implementation.

```
export BYTEPS_NCCL_NUM_RINGS=z
```

BytePS uses group NCCL calls to reduce NCCL invoking overhead. You can try to increase the group sizes:

```
export BYTEPS_NCCL_GROUP_SIZE=w
```

Servers can also be the performance bottleneck, e.g., when there are only one server but multiple workers.
You can try to increase the number of processing threads on the servers (default is 4):

```
export BYTEPS_SERVER_ENGINE_THREAD=v
```

Or enable scheduling at the server side to prioritize tensors with higher priority:

```
export BYTEPS_SERVER_ENABLE_SCHEDULE=1
```

## Asynchronous training

Enable asynchronous training with (on all workers and servers)

```
export BYTEPS_ENABLE_ASYNC=1
```

## Core affinity

`BYTEPS_NUMA_ON`: Enable or disable core affinity. Valid values 0 or 1, default value is 1.

`BYTEPS_NUMA_DEFAULT_QUOTA`: The number of physical cores to assign to a worker. It's default value is calculated:

        `total number of physical cores / number of byteps workers`

`BYTEPS_CPU_BLACKLIST` a string of comma-separated logical CPU core ids to avoid using. Only has effect in the automatic core assignment mode.

`BYTEPS_VISIBLE_CPU_CORES`: if this envar is set, it overrides the automatic core assignment. It's a colon-separated string of logical core ids, each section is a list of cores for a byteps worker. The number of sections should match the number of byteps workers launched.
    For example, there are 2 byteps workers on a node, rank 0 is tied to GPU 3, rank 1 is tied to GPU 7. And we know cores 1,4-5,7-11,12 are one the same numa node as GPU 3, and cores 20-25 are on the same numa node as GPU 7. Then we could use:  
    `export BYTEPS_VISIBLE_CPU_CORES=1,4-5,7-11,12:20-25`  
    to assign cores 1,4-5,7-11,12 to rank 0, cores 20-25 to rank 1.

`BYTEPS_MULTITHREADED_CPU`: 0 means hyperthreading is disabled. 1 means hyperthreading is enabled. The default value is 1. If its enabled both logical cores on a physical core will be used. Assumption: if there are `n` physical cores, the two logical cores on physical core `i` are logical core `i` and `i + n`.

The automatic core assignment assigns physical cores to byteps workers such that:
  1) one worker's cores are on the same numa node with its GPU
  2) one worker's cores don't overlap with another worker's cores
  3) if hyperthreading is enabled, assign logical cores to the worker as well.
