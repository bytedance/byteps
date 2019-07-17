# Troubleshooting

We suggest you read the Horovod troubleshooting, especially for problems during the build process. BytePS has almost the same dependencies as Horovod minus MPI.

https://github.com/horovod/horovod/blob/v0.16.4/docs/troubleshooting.rst


## MXNet coexistence

BytePS allows you to use any MXNet version>=1.4.0 as the worker. However, if you install the MXNet worker package in the same environment/docker as BytePS server/scheduler, you must make sure that BytePS server/scheduler imports the MXNet built specifically for BytePS. You can set the path:

```
BYTEPS_SERVER_MXNET_PATH=/path/to/your/mxnet/for/byteps/server
```

Your worker's MXNet, which is installed via pip, for example, is placed in the default python searching path. Thus, on workers you do not need to set this.

A better way is to avoid installing MXNet worker package and BytePS server in the same image, i.e., use different images for worker and server/scheduler.


## Network connectivity

When launching distributed jobs, if you see hanging at the beginning, one possible reason is that your network connection has trouble. You can use `ps-lite` benchmark to verify the connectivity.

Install ps-lite:

```
git clone --branch byteps https://github.com/bytedance/ps-lite.git
cd ps-lite
make -j
``` 


For the scheduler
```
export DMLC_ROLE=scheduler 
export DMLC_NUM_WORKER=1 
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=[YOUR_SCHEDULER_IP] 
export DMLC_PS_ROOT_PORT=[YOUR_SCHEDULER_PORT] 
export DMLC_INTERFACE=eth0 
./ps-lite/tests/test_kv_app_benchmark 
```

For the server
```
export DMLC_ROLE=server 
export DMLC_NUM_WORKER=1 
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=[YOUR_SCHEDULER_IP] 
export DMLC_PS_ROOT_PORT=[YOUR_SCHEDULER_PORT] 
export DMLC_INTERFACE=eth0 
./ps-lite/tests/test_kv_app_benchmark 
```

For the worker:
```
export DMLC_ROLE=worker 
export DMLC_NUM_WORKER=1 
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=[YOUR_SCHEDULER_IP] 
export DMLC_PS_ROOT_PORT=[YOUR_SCHEDULER_PORT] 
export DMLC_INTERFACE=eth0 
./ps-lite/tests/test_kv_app_benchmark 1024000 100 0
```

If it succeed, you should be able to see something like this on the worker. 
```
tests/test_kv_app_benchmark.cc:77: push_byte=4096000, repeat=100, total_time=128.842ms
tests/test_kv_app_benchmark.cc:91: pull_byte=4096000, repeat=100, total_time=353.38ms
```

(Note: for RDMA networks, use `make -j USE_RDMA=1` to build, and `export DMLC_ENABLE_RDMA=1` for running the scheduler / server / worker)

If it still hang, you may need to check your network connectivity.