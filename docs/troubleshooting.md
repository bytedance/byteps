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
