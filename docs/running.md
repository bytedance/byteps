# Running BytePS

BytePS follows the same running model as MXNet's PS implemenation, and provides a script, launcher/launcher.py, to help you start individual processes. **Below instructions, including those DMLC variables, apply to all frameworks.**

Let's say you have two worker machines (or docker containers) that have GPUs, one machine or container as a server, and a scheduler. The scheduler binds on 10.0.0.1 and port 9000. The workers and the server can connect to the scheduler via the IP and port using TCP.

To use launcher/launcher.py, NVIDIA_VISIBLE_DEVICES should exist -- either automatically set by nvidia-docker, or manually set by you.

On worker 0, run:

```
DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.0.0.1 DMLC_PS_ROOT_PORT=9000 \
DMLC_WORKER_ID=0 DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 \
bpslaunch YOUR_COMMAND
```

On worker 1, run (only DMLC_WORKER_ID is different from above):

```
DMLC_ROLE=worker DMLC_PS_ROOT_URI=10.0.0.1 DMLC_PS_ROOT_PORT=9000 \
DMLC_WORKER_ID=1 DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 \
bpslaunch YOUR_COMMAND
```

**For servers and schedulers, we highly recommend you use the docker image we build:**

```
docker pull bytepsimage/byteps_server
```

Start server and scheduler docker instances with this image. In the server, run the following. Compared with the worker command, we remove DMLC_WORKER_ID, and set role to server.

```
DMLC_ROLE=server DMLC_PS_ROOT_URI=10.0.0.1 DMLC_PS_ROOT_PORT=9000 \
DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 bpslaunch
```

On the scheduler, run (we also remove DMLC_WORKER_ID, and set role to scheduler):

```
DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=10.0.0.1 DMLC_PS_ROOT_PORT=9000 \
DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 bpslaunch
```

In this example, your scheduler must be able to bind to `10.0.0.1:9000`.

The order of starting workers/servers/scheduler does not matter. 


## Gradient Compression 

Gradient compression is an optional feature in BytePS. The scalability of distributed training is severely hindered by communication overhead. One way to reduce the overhead is to perform lossy compression to gradients, i.e., gradient compression. If your workload is communication-intensive (e.g., large model), you can consider enabling this feature.  We provide some state-of-the-art algorithms to use, including 1-bit, top-k, random-k, and dithering. 

To use it, you only need to add a few lines of code in your script. Specifically, which algorithm to use and its corresponding parameters are required to be passed to the BytePS's optimizer through a python dictionary. For example,

```python
compression_params = {
    "compressor": args.compressor,
    "ef": args.ef,
    "momentum": args.compress_momentum,
    "scaling": args.scaling,
    "k": args.k,
    "normalize": args.normalize,
    "partition": args.partition,
    "fp16": args.fp16_pushpull
}

# MXNet
trainer = bps.DistributedTrainer(
    params, "sgd", optimizer_params, compression_params=compression_params)


# PyTorch
optimizer = bps.DistributedOptimizer(optimizer,
                                    named_parameters=model.named_parameters(),
                                    compression_params=compression_params)
```


### References

|name | valid input| note| 
|---- | ---- | --- |
|compressor| "onebit", "topk", "randomk", or "dithering"| algorithm |
|ef | "vanilla", "corrected" and "sparse" | error-feedback, "sparse" is only valid for "randomk" |
|momentum| "nesterov" | added before compression |
|scaling| True or False| only valid for "onebit" |
|k| float or integer | only valid for "topk", "randomk", and "dithering"| 
|normalize| "l2" or "max"| only valid for "dithering" |
|partiton| "linear" or "natural" | only valid for "dithering" |
|fp16| True or False| fp16 pushpull|