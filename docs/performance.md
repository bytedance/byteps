# BytePS Performance with 100Gbps RDMA

Note: here we present the *worse case scenario* of BytePS, i.e., 100Gbps RDMA + no NVLinks. 

We get below results on machines that are based on PCIe-switch architecture -- 4 GPUs under one PCIe switch, and each machine contains two PCIe switches.
The machines are inter-connected by 100 Gbps RoCEv2 networks.
In this case, BytePS outperforms Horovod (NCCL) by 7% for Resnet50, and 17% for VGG16. 

<img src="/docs/images/perf_rdma_vgg16.png" width="360" height="220"><img src="/docs/images/perf_rdma_resnet50.png" width="360" height="220">

To have BytePS outperform NCCL by so little, you have to have 100Gbps RDMA network *and* no NVLinks. In this case, the communication is actually bottlenecked by internal PCI-e switches, not the network. BytePS has done some optimization so that it still outperforms NCCL. However, the performance gain is not as large as other cases where the network is the bottleneck.

As long as you have NVLinks, or you run on slower networks, the performance gain of BytePS will be closer to [README.md](/README.md).
