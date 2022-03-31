# BytePS Architecture

We highly recommend you to read [BytePS's rationale](./rationale.md) first before reading this doc.

From application views, BytePS is a communication library just like Horovod. The plugins handle framework-specific transformation (e.g., on data structure), and
put communication tasks into BytePS priority queues. The BytePS Core then gets the tasks (priority-aware, not FIFO) and handles the actual communication.

![byteps_architecture](https://user-images.githubusercontent.com/13852819/69873605-c3d39e00-12f3-11ea-942d-97af2606bb40.png)


## General Workflow
To demonstrate the work flow of BytePS, below we use a common data-parallel training scenario as an example. Say we have multiple worker machines (we refer them as "**workers**"), and each machine (worker) has multiple GPUs. We also have some CPU machines that serve as PS (we refer them as "**servers**").

In BytePS, a general walk-through of an iteration goes like this (we call each step as a **stage**):

1. **Computation**: Each GPU performs computation (forward/backward propagation), which is irrelevant to BytePS;
2. **Local Reduce**: Multiple GPUs on the same machine reduces the gradients;
3. **Push**: The workers push the aggregated gradients to the servers;
4. **Global Reduce**: Once the servers receive the gradients from different workers, it aggregates the gradients;
5. **Pull**: The workers pull the aggregated gradients from the servers;
6. **Local Broadcast**: The workers broadcasts the updated gradients to local GPUs;
8. Goto next iteration and repeat from 1.


## Local Communication

We use NCCL for local communication, including **Local Reduce** and **Local Broadcast**.

For **Local Reduce** stage we use [ReduceScatter](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#reducescatter) to evenly distribute the gradients on multiple GPUs.

For **Local Broadcast** stage we use [AllGather](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allgather) to broadcast the gradients back to multiple GPUs.

## Distributed Communication

We use [ps-lite](https://github.com/bytedance/ps-lite/tree/byteps) for **Push** and **Pull** between workers and servers.

For **Push** stage, the workers send the gradients to servers, as the traditional PS does.

For **Pull** stage, the workers <u>pull gradients rather than parameters</u> from the servers, which is different from traditional PS. Here is why:

In past, the SGD update is performed on servers, so the workers need to tell the servers what SGD optimizer to use. However, for different frameworks, even the same optimizer algorithm may be implemented in completely different ways, and not to mention there are many user-defined optimizers. So BytePS moves the SGD update from the servers to the workers, leaving the servers only do gradient reduction. We believe this is generic because it applies to all frameworks we know so far.



