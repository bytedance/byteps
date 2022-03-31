# FAQ about PS vs Allreduce

Below we summarize a list of questions or *incorrect* statements that many users are confused of.

### **BytePS has a better performance because of some data path details, e.g., less copying?**

Not really. BytePS uses share memory in a similar way as NCCL, and BytePS copies the data for as many times as NCCL. In addition, both BytePS and NCCL use tensor partition/segmentation internally, which hides most of the copying delay.

I would consider BytePS and NCCL are both close to their theoretical optimal based on their communication patterns.

### **BytePS has a better performance because it has a hierarchical strategy, i.e., local reduce followed by inter-machine transfers?**

Hierarchical strategy does help a bit. However, it is not as fundamental as the PS communication pattern. Consider the following example:

You have N worker machines connected to the same network switch, and each worker has only 1 GPU. In this case, the topology is flat -- no hierarchy at all. In this case, you'll find the analysis in [rationale.md](/docs/rationale.md) still applies. PS communication pattern has less traffic volume *from worker's stand point of view*.

### **Allreduce is equivelant to PS, as long as you find the correct allreduce strategy?**

Not really. Consider the above flat N-worker example again. No matter with which allreduce strategy, the conclusion of PS vs. allreduce does not change.

### **Okay, I get that PS has less traffic from workers. But, PS Push and Pull are not duplex, and waste half bandwidth?**

BytePS does not have this problem. It can fully utilize bi-direction network bandwidth. The key ideas are tensor partitioning and pipelining. For example, you have a 100MB tensor to be pushed and pulled. Inside BytePS, we will partition the tensor into small pieces. After pushing the first piece, we will start pulling the first piece. At the same time, we will start pushing the second piece. And so on. For most of the time except the first piece and the last piece, the bi-directional bandwidth is fully utilized.

### **Since the bottleneck is the NIC of GPU machines, why not add more NICs?**

There are indeed [specialized physical server designs](https://images.nvidia.com/content/pdf/dgx1-v100-system-architecture-whitepaper.pdf) doing that. Unfortunately, cloud or shared clusters usually prefer not do this. This is because, as a matter of fact, many training jobs are not distributed. For these jobs, users want the GPUs to be deployed as dense as possible and the network bandwidth requirement is low.

If you are building your own cluster for a *single* dedicated *distributed* training job, of course you can go with the HPC route, carefully calculate the best ratio between GPUs and NICs, build a homogeneous cluster and use allreduce. However, please realize that cloud and shared clusters are not HPC. This is the whole point of BytePS.

### **Does PS architecture impose heavier cross rack traffic, and may be impacted by physical network over-subscription ratio?**

This is true. For a large job that workers and PS cannot fit side a rack, PS does have more cross-rack traffic.

However, the comparison with allreduce in real life is more complicated. It depends on how well you can control the physical job placement and allreduce rings. If you don't have the full control of placement, or your MPI/NCCL rank assignment is not physical network topology-aware, allreduce would face the exactly same problem. NCCL and most MPIs today are unaware of physical network topology, unless specifically designed for a given HPC.

Don't be scared of the oversubscription ratio. It exists for a reason -- usually, not all servers in a rack are simultaneously busy on networking. Multiple researches from major cloud providers show that the average bandwidth utilization is low. Remember, this is a shared cluster, not everyone is running distributed training.

### **Final remarks**

With BytePS, we want to share two key insights --

* Cloud, either public or private, is different from HPC. Using ideas from HPC is a shortcut, but not optimal.
* In a (public or private) cloud, PS architecture is theoretically better than allreduce, with minimal additional costs.

BytePS is a realization of the idea.
