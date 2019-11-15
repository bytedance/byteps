# BytePS Performance with 100Gbps RDMA

## NVLink + RDMA

We show our experiment on BERT-large training, which is based on GluonNLP toolkit. The model uses mixed precision.

We use Tesla V100 32GB GPUs and set batch size equal to 64 per GPU. Each machine has 8 V100 GPUs with NVLink-enabled. 
Machines are inter-connected with 100 Gbps Infiniband network. 

BytePS outperforms Horovod (after carefully tuned) by 16% in this case, both with RDMA enabled.

![perf_rdma_nvlink](https://user-images.githubusercontent.com/13852819/68922123-cb545c80-07b5-11ea-884b-7d541a848031.png)

## PCIe + RDMA

Note: here we present the *worse case scenario* of BytePS, i.e., 100Gbps RDMA + no NVLinks. 

We get below results on machines that are based on PCIe-switch architecture -- 4 GPUs under one PCIe switch, and each machine contains two PCIe switches.
The machines are inter-connected by 100 Gbps RoCEv2 networks.
In this case, BytePS outperforms Horovod (NCCL) by 7% for Resnet50, and 17% for VGG16. 

![perf_rdma_pcie_resnet50](https://raw.githubusercontent.com/bytedance/byteps/master/docs/images/perf_rdma_resnet50.png)

![perf_rdma_pcie_vgg16](https://raw.githubusercontent.com/bytedance/byteps/master/docs/images/perf_rdma_vgg16.png)


To have BytePS outperform NCCL by so little, you have to have 100Gbps RDMA network *and* no NVLinks. In this case, the communication is actually bottlenecked by internal PCI-e switches, not the network. BytePS has done some optimization so that it still outperforms NCCL. However, the performance gain is not as large as other cases where the network is the bottleneck.

As long as you have NVLinks, or you run on slower networks, the performance gain of BytePS will be closer to [README.md](/README.md).
