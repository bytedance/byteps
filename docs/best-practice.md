# BytePS Best Practice

## Single machine (non-distributed mode)

When `DMLC_NUM_WORKER=1`, BytePS will not use the parameter servers or schedulers at all. In this case, BytePS runs in non-distributed mode. You do not even need to run server/scheduler.

In non-distributed mode, BytePS is basically doing NCCL allreduce, so it will not outperform Horovod/NCCL much. BytePS implemented priority-based scheduling, which may improve the training speed by 0%~15%, depending on your training task.

The only thing you can tune is `BYTEPS_PCIE_SWITCH_SIZE`. If you know your hardware topology, e.g., say you have 8 GPUs in total, 4 GPUs connect to one PCI-e switch, the other 4 GPUs connect to another PCI-e switch, then you should set `BYTEPS_PCIE_SWITCH_SIZE=4`. In this case, you may see 20%~30% performance improvement compared with Horovod/NCCL.

If you have NVLinks, leave `BYTEPS_PCIE_SWITCH_SIZE` unmodified. If you don't know your hardware topology, leave `BYTEPS_PCIE_SWITCH_SIZE` unmodified.


## Multi-machine (distributed mode)

### With additional CPU servers

This mode requires at least **4** physical machines. Two of the machines should have GPUs and run as workers. The other two run as CPU servers and do not need GPUs. The scheduler can run on any machine.

The key here is to make sure the following:
* Servers must be on different physical machines from workers.
* The total bandwidth of the servers must be equal or larger than the total bandwidth of workers.

If you are using RDMA, this should be sufficient. However, with TCP and >=25Gbps networks, it's possible that BytePS cannot fully utilize the bandwidth because a single TCP connection usually cannot run up to 25Gbps.

To address this, you can try running more BytePS server instances on the server machines. For example, you can try running two server instances per server machines. This effectively doubles the number of TCP connections and should be sufficient for 25Gbps networks. For 40Gbps/50Gbps networks, you need three server instances per server machine, and so on.

### No additional CPU servers

When you don't have additional CPU servers, then for each physical machine, you should launch a worker and a server process. We call this *co-locate* mode, and the resource consumption is the same with Horovod (no additional servers).

If you are using TCP, you will probably get near-identical performance with Horovod-TCP. However, if you are using RDMA, you can set `BYTEPS_ENABLE_IPC=1` to enable the IPC communication between the co-located worker and server. And eventually you will get higher end-to-end performance than Horovod.

## The expected performance

In the single machine case, if you leave `BYTEPS_PCIE_SWITCH_SIZE` unmodified, BytePS performance should never be lower than Horovod/NCCL.

In multi-machine case, if the deployment satisfies the two requirements above, you should see BytePS is at least as fast as Horovod or TF and MXNet's native PS. If each of your workers has two or more GPUs, you should see significant improvement, like 40% - 100% compared with other existing solutions.

If you have to deploy server instances on the same physical machines as workers, the performance will be similar to Horovod/NCCL.

If you have less servers than workers, the performance will be proportionally lower. For example, if you have only 1 server and 2 workers, you'll only get half of the performance compared with 2 servers + 2 workers.

## How to compare with other solutions

To compare with Horovod is simple. Install Horovod, and change `bps` back to `hvd`.

To compare with other PS architecture, make sure that you use the same hardware setup. Most of the existing PS implementations cannot run as fast as Horovod/NCCL. So, usually you just need to compare with Horovod/NCCL.
