# The Rationale of BytePS

We find that some users may not fully understand this page. If you have doubts after reading this page, we prepare a detailed [FAQ](/docs/faq.md). If you still have questions, you are welcome to raise Github issues.

## Background

You want to run your training in your expensive GPU cluster. Unfortunately, when you run distributed training, these GPUs are not well fed -- their precious cycles are wasted on waiting for network transmission. You tried many existing solutions, most notably the popular allreduce approach. You probably found NCCL gave you better performance than many other alternatives. You know NCCL is developed by NVIDIA and HPC experts. You guess that may be it.

We understand you. This is why we develop BytePS and want to show you, in a cloud or in-house shared cluster environment, NCCL (and fundamentally allreduce) is suboptimal.

BytePS offers you an option: if you invest 2% more in network and CPU, BytePS lets you achieve 40% - 100% faster GPU training, what would you do? Further, what if you don't actually invest more, but to utilize the spare network and CPU resources in your shared clusters?

## The PS communication pattern is better, theoretically

Let's review your system architecture. You have beast machines whose PCI-e are mostly occupied by GPUs. GPUs are the most expensive part in your system. During distributed training, the communication time wastes GPU cycles. What determines the communication time, given a certain amount of data to send? It is the bandwidth bottleneck somewhere in your system.

In most system setups, this bottleneck is the network. More precisely, it is the network interface card (NIC) on your GPU servers. There are many reasons -- many networks are slower than PCIe 3.0 x16 that your GPUs use. The PCIe lanes on a machine are mostly allocated to GPUs, not your NICs. NVLinks, if you have them, are even faster than PCIe, let alone your NICs.

After the analysis, you'll find that the speed of distributed training depends on how effectively you use the NIC bandwidth.

Let's check allreduce first. [NCCL document](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) summarizes it well -- to allreduce 1MB of data, you need to send and receive `(2*(n-1)/n)` times 1MB across every link in your topology. In naive NCCL, this `n` is the number of GPUs. With some hierarchical strategy, this `n` may be reduced to the number of machines. Nevertheless, with larger `n`, you can see that `(2*(n-1)/n)` will quickly go towards 2.

Now we check the Parameter Server (PS) communication pattern. For 1MB data, every worker only has to send this 1MB *once*, and (after the PS sums up the data across the workers) receive 1MB *once*. This means, with PS architecture, your bottleneck bandwidth is utilized up to twice more efficient than allreduce!

## Wait. Why did MPI guys not find out this?

Well, MPI guys probably know this, but they might be too focused on homogeneous hardware setup, like what HPC has. The trick of PS here is that you need additional resources, other than the GPU workers, to run the PS. If all your machines are homogeneous, e.g., equipped with 8 GPUs and same CPUs, memory and NICs, and all your machines are only for a single job, then PS does not save you anything. In that case, PS architecture would require you to use only half GPU machines as workers, and the other half as PS, so that the network bandwidth on PS side matches worker side. This alone will make you waste half GPUs. So, in a homogeneous setup designed for a single job, allreduce is your best bet.

However, the real data centers are a bit different. Inside a large organization, we usually run a large-scale GPU cluster and provide it as a service to multiple teams for multiple jobs. The GPU clusters are usually connected to other parts of the data center, where there is an enormous pool of CPUs and network bandwidth. The same goes for public cloud.

In light of this, BytePS is specifically designed to run PS instances using only CPUs, while workers run on GPU machines.

## CPU-only PS: small costs, huge gains

GPUs are extremely expensive compared with CPUs and network bandwidth. Use AWS public price sheet as an example. If you rent 4x [p3.16xlarge](https://aws.amazon.com/ec2/instance-types/p3/), it would cost you nearly $100 per hour. However, to match the network bandwidth, you can rent 4x or even 8x [c5n.xlarge](https://aws.amazon.com/ec2/pricing/on-demand/) as your PS for $0.2 per instance per hour!
c5n.xlarge has 4 CPU cores that are sufficient for BytePS and its [up to 25Gbps](https://aws.amazon.com/ec2/instance-types/) network.

Therefore, on a public cloud, you just need 2% more spending, you may get up to 100% improvement because your bottleneck bandwidth is utilized twice as efficient. If you manage your own training cluster, you may not really have any additional spending, because you probably have spare CPU and networking resources somewhere in your data center.

## I still don't understand. NCCL beats TF (or MXNet) PS so hard..

The poor performance of the original PS implementation of Tensorflow and MXNet has many reasons -- poor engineering, did not overlap computation and communication well, did not utilize bidirectional network bandwidth well, did not handle local merging across local GPUs well, did not support RDMA, etc.

BytePS solves all these problems for you. If you are curious about the implementation details, read the code. We will also release a detailed technical paper soon.

## One PS for all

You may ask, isn't allreduce more generic than PS? Horovod works for TF, PyTorch, MXNet and Keras. There was not a PS implementation that is as generic.

BytePS has an answer. BytePS shows that, with Horovod-alike interfaces, PS communication pattern can be as generic. The current BytePS works for TF, PyTorch, MXNet and Keras, and we do not see any fundamental reasons that there may be any framework that can run Horovod but cannot run BytePS.

Well, to achieve this, BytePS does not work strictly the same as original PS design. We will explain this in other parts of the documentation.

## Other benefits

Finally, there are many other benefits of using PS over allreduce. For example, it does not have the synchronization cost when you train on large-scale. Allreduce requires all workers (GPUs) to have a global barrier before each allreduce. PS does not have this problem. Workers are asynchronous by their nature.

Furthermore, it is much easier to support asynchronous training with PS (we will have detailed paper talk about this later) than allreduce. It is also easier to add fault-tolerance, straggler-mitigation, etc. We have not implemented them all, but we plan to, and we welcome contributions. Any such features added to BytePS will benefit all supported framework. Isn't this great?
