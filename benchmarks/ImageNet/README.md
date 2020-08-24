# Benchmarking Guide


- [Benchmarking Guide](#benchmarking-guide)
  - [Step-by-Step Tutorial](#step-by-step-tutorial)
  - [Necessary Configurations](#necessary-configurations)
    - [Network](#network)
    - [GPU](#gpu)
  - [Debug](#debug)

## Step-by-Step Tutorial

Given that you have N nodes, each equipped with 8 GPUs and ethernet interconnection. In this guide, we take training ImageNet with ResNet50 for example. Now let's begin our tour of distributed training with BytePS.

0. make sure the first node can ssh to other machines without passwords.



1. install BytePS in all your machines. 

Please refer to our `install guide`. 

2. download training scripts in all your machines. 

We recommend you just clone the `byteps` repo in your machines. 

For this experiment, 

3. 




## Necessary Configurations

**Note that you may need to change some default configurations to accomodate to your own machines.** The default configurations are set according to Amazon's P3 instances.

### Network

1. network interface

To check which network interfaces you have, please commnad `ip link show`. For example,

```sh
$ ip link show
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: ens3: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9001 qdisc mq state UP mode DEFAULT group default qlen 1000
    link/ether 16:6d:af:e6:fb:03 brd ff:ff:ff:ff:ff:ff
3: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN mode DEFAULT group default
    link/ether 02:42:86:8c:43:7c brd ff:ff:ff:ff:ff:ff
```

- `lo` - Loopback interface 
- `ens3` - Your ethernet network interface at PCIe 
- `docker0` - docker interface

If your network interface is not `ens3`, please replace it with your own ethernet network interface (e.g. `eht0` ) in the [train_imagenet.sh](train_imagenet.sh).

For more about interface naming, please check [Consistent Network Device Naming](https://en.wikipedia.org/wiki/Consistent_Network_Device_Naming).

2. port 

The default port is 1234. Please make sure it is not taken by other processes. You can check it with `lsof`. 

```sh
$ lsof -i:1234
```

If there is no output, then the port is unused. You can also change the default port in the [train_imagenet.sh](train_imagenet.sh).

### GPU

You need to determine NVIDIA_VISIBLE_DEVICES based on the number of GPUs in your machine. For example, if the machine has 8 GPUs, you should set `NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` in the [train_imagenet.sh](train_imagenet.sh).figurations

**Note that you may need to change some default configurations to accomodate to your own machines.** The default configurations are set according to Amazon's P3 instances.


## Debug

