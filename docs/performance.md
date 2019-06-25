# BytePS Performance with RDMA

We further evaluate BytePS on RDMA networks. The machines are based on PCIe architecture (4 GPUs under one PCIe switch), and each machine contains two PCIe switches.
The machines are inter-connected by 100 Gbps RoCEv2 networks.
In this case, BytePS outperforms Horovod by 7% for Resnet50, and 17% for VGG16. 

<img src="/images/perf_rdma_vgg16.png" width="360" height="220"><img src="/images/perf_rdma_resnet50.png" width="360" height="220">

