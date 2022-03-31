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
