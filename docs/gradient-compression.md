## Motivation

Currently BytePS does not fully support gradient compression. The compression it supports lies in each plugin in Python. Such design may ease the difficulty of the implementation but leads to major inabilities for more aggressive compression. This is because NCCL only supports limited reduction operations such as Sum, Prod etc but these operations are meaningless for the compressed data which have been highly bit-wisely packed. For example, for [signSGD](https://arxiv.org/pdf/1802.04434.pdf),  one of the most popular methods for gradient compression due to its simplicity and effectiveness, each bit represents a signbit of an element in the original data tensor, making reduction operations like summation totally meaningless. But reduction is necessary for multi-GPU devices.

Another problem is that compared to inter-node communication, intra-node communication is not the bottleneck. Furthermore, too much compression at first will lose much information, which may cause low accuracy. So there is no need to make too radical compression before running into BytePS core in worker nodes.

Therefore, changes need to be made.

## Design Overview

In light of the problems mentioned above, we propose two-level gradient compression:

1. intra-node: This is just an alias for the current implementation, named after its communication property. Transform FP32 tensors into FP16 on each GPU, reduce them across multi-GPUs via NCCL, and copy them to the CPU buffer waiting for next-level compression. The purpose of the compression is to reduce intra-node communication overhead introduced by multi-GPUs. Since intra-node communication is very fast, especially with NCCL, only mild compression methods will be applied, most of which is type-conversion. It is framework-specific and will be implemented in each plugin.

2. inter-node: Usually inter-node communication is a bottleneck, so more drastically gradient compression algorithms will be applied here. This is framework-agnostic and will be implemented in BytePS core.

It is worth mentioning that our design supports all frameworks. 

![architecture](https://user-images.githubusercontent.com/25879526/86322951-7abf4000-bc6e-11ea-871f-572a7efed7cd.png)

## Interface

Only a few changes to be made for users. Users only have to add a few LOC in the script to specify which compression algorithm to be used and the parameters needed by the algorithm. Take MXNet for example. 

```python
compression_params = {
            "compressor": opt.compressor,
            "ef": opt.ef,
            "momentum": opt.compress_momentum,
            "scaling": opt.onebit_scaling,
            "k": opt.k
}

trainer = bps.DistributedTrainer(params, optimizer, optimizer_params, compression_params=compression_params)
```

Here we prescribe some keys. Users can lookup documentations to determine which key should be used. Here are some common keys.

| KEYS | DESC |
| --- | --- |
| compressor | compression algorithms, including onebit / dithering / topk / randomk |
| k | an integer, must be specified when using dithering / topk / randomk |
| scaling | optional, whether to enable scaling for onebit, default is false |
| ef | error-feedback algorithms, e.g. vanilla |
| momentum |  momentum algorithms, e.g. nesterov  |
| seed |  random seed  |

If the user's input is not correct, it will give a warning and abort.

## Implementation

### Parameter Data Structure

To offer users a unified interface to use,  we have to address the registration problem. parameters vary from different kinds of compression algorithms. For example, topk and randomk algorithms need parameter k to be specified while onebit algorithm may need to input whether to enable scaling flag. Some parameters are optional but others are not. So parameter passing is a challenge.

We address this challenge using _string-string dictionary_ (`std::unorded_map<std::string, std::string>` for C++ or `dict` for Python) as our unified data structure to pass parameters. As mentioned above, we prescribe specific strings as keys, so the _dictionary_ will look like:

```python
{"byteps_compressor_type": "topk", "byteps_compressor_k": "3", "byteps_error_feedback_type": "vanilla"}
```

**Python**

For MXNet users, the dictionary can be an attribute of ParameterDict. We can filter out those parameters by leveraging the prefix "byteps". For example,

```python
for i, param in enumerate(self._params):
           byteps_declare_tensor("parameter_" + str(i))
           if param.grad_req != 'null':
               byteps_params = dict(
                   filter(lambda attr: attr[0].startswith(
                       "byteps_",), param.__dict__.items())
               )
               byteps_declare_tensor("gradient_" + str(i), **byteps_params)
```

**C++**

Using ctypes, we can pass the dictionary conveniently. For example,
```c++
extern "C" void byteps_mxnet_declare_tensor(char* name, int num_params,
                                           char** param_keys,
                                           char** param_vals) {
 ...

 std::unordered_map<std::string, std::string> param_dict;
 std::string key, val;
 std::string::size_type pos;
 for (int i = 0; i < num_params; ++i) {
   key = param_keys[i];
   val = param_vals[i];
   param_dict[key] = val;
 }

 ...
}
```

### Compressor - Development API

We want developers to develop their own gradient compression algorithms without fully understanding how BytePS works. What they only need to know is development API. We currently implement some commonly used gradient compression algorithms, but in the future, we hope more novel algorithms will be implemented under our API. We abstract compression algorithms into `compressor`. The `Compressor` looks like this:

```c++
class Compressor {
 public:
  Compressor(size_t size, DataType dtype)
      : _size(size),
        _dtype(dtype),
        _buf(new byte_t[size]),
        _cpu_reducer(new CpuReducer(nullptr)){};
  virtual ~Compressor() = default;

  virtual tensor_t Compress(tensor_t grad) = 0;

  virtual tensor_t Decompress(tensor_t compressed) = 0;

  virtual void FastUpdateError(tensor_t error, tensor_t corrected,
                               tensor_t compressed) {
    BPS_LOG(FATAL) << "FastUpdateError is not implemented";
  };

  std::unique_ptr<byte_t[]> _buf;

  size_t _size;

  DataType _dtype;

  std::unique_ptr<CpuReducer> _cpu_reducer;
};
```

In order to make less modifications to BytePS core, we want compressors to be as general as possible. In the best case, the base compressor pointer/reference can represent all kinds of compressors and only need to expose two operations to users: `Compress` and `Decompress`. This is quite challenging because there are some optional features for gradient compression, such as error-feedback and momentum. These are two common methods to correct the bias and accelerate the training process respectively. For example, with error-feedback, before being compressed, gradients are first corrected with errors which refer to the information loss during the last compression,  and then errors are re-calculated. Therefore, the workflow is different from only using vanilla gradient compression.

In order to support all these features and expose a unified API at the same time, we use the decorator pattern. We regard error-feedback as an additional behavior of compressors. We want a unified API, which means compressors with error-feedback should expose the same method as those without error-feedback. But in that case we have to create a subclass for each compressor, which is too redundant. So the decorator pattern just solves our problem. We create a decorator class named `ErrorFeedback` to inherit `BaseCompressor` while at the same time also keeping a member of `BaseCompressor`. For example,

```c++
class ErrorFeedback : public Compressor {
 public:
  ErrorFeedback(size_t size, DataType dtype, std::unique_ptr<Compressor> cptr)
      : Compressor(size, dtype),
        _cptr(std::move(cptr)),
        _error(new byte_t[size]()) {}
  virtual ~ErrorFeedback() = default;

  virtual tensor_t Compress(tensor_t grad) final;

  virtual tensor_t Decompress(tensor_t compressed) final;

 protected:

  virtual void UpdateGradient(tensor_t grad) = 0;

  virtual void UpdateError(tensor_t corrected, tensor_t compressed);

 protected:
  std::unique_ptr<byte_t[]> _error;

 private:
  std::unique_ptr<Compressor> _cptr;
};
```

And the workflow is implemented in `Compress` and `Decompress`. For example,
```c++
tensor_t ErrorFeedback::Compress(tensor_t grad) {
  // 1. grad <- grad + error
  UpdateGradient(grad);

  // 2. c <- Compress(grad)
  auto compressed = _cptr->Compress(grad);

  // 3. e <- grad - Decompress(c)
  UpdateError(grad, compressed);

  return compressed;
}

tensor_t ErrorFeedback::Decompress(tensor_t compressed) {
  // directly forward to internal compressor
  return _cptr->Decompress(compressed);
}
```

`Momentum` is implemented in the same way. `ErrorFeedBack` and `Momentum` are also base classes to inherit. In this way, error-feedback and momentum becomes optional features to be added to any vanilla gradient compression algorithms.

BTW, momentum is not applied to servers. 

## Exps

### CIFAR100

#### End-to-End Training

We conduct the experiment in distributed training ResNet18_v2 on the CIFAR100 datasets with 4 AWS P3.16xlarge instances, each equipped with 8 V100 GPUs and 25Gbps network. The compression algorithms benchmarked here are also equipped with error-feedback and nesterov momentum. We set k = 1 for topk and k = 8 for randomk. We train it for 200 epochs. 

![image](https://user-images.githubusercontent.com/25879526/86323315-38e2c980-bc6f-11ea-9c5c-038371d5d6b5.png)

![image](https://user-images.githubusercontent.com/25879526/86323299-2ec0cb00-bc6f-11ea-82d8-ee31c4bb3ec8.png)

|  f888c8d8f9e8483e46acd00042ed262e30c6856e  | VAl ACC | TIME(s) |
| -- | -- | -- |
|baseline| 0.713799| 703.1527987500002|
|onebit| 0.705601| 629.4210848750001|
|randomk| 0.6991| 501.99770550000005|
|topk| 0.704202| 507.90769437499966|


The results show that compression can reduce up to 28.6% end-to-end training time without accuracy loss. 

#### Slow Network

Gradient compression is more beneficial in slower network. Therefore we limit the network bandwidth to 100Mbps (both downlink and uplink) and keep all other settings not changed. The results show that we can achieve up to 6x reduciton in training time. 

![image](https://user-images.githubusercontent.com/25879526/86326780-c96fd880-bc74-11ea-9bcf-673f061f0020.png)

|  b382f996d159fbe4d48c1135290f5c4183fc6b46  |  TIME(s) |
| -- | -- |
|baseline| 518.321322125|
|onebit| 195.236724875|
|randomk| 89.672168625|
|topk| 83.9287285|

### IMAGENET

To save time, we only tested 1bit algorithm. Topk and randomk are not guaranteed to converge on IMAGENET. 

#### Workload Breakdown 

In this experiment, we measure the workload breakdown into computation and communication. We use 8 Amazon EC2 p3.2xlarge instances, each of which is shipped with one Nvidia V100 GPU and 10Gbps Ethernet. We train two CNN models: Resnet-50_v2  and VGG-16. We first measure the computation time by collecting the elapsed time of running 50 iterations (t0) on one node. Then we measure the total training time for running 50 iterations (t1) on 8 nodes. Then, we get an estimate of communication time using t1 − t0.

As the figure shows, dist-EF-SGDM can reduce communication to varying degrees. For ResNet50_v2, the drop is trivial (17.6% decrease), mainly due to the smaller model size. In contrast, a remarkable decline (73.2% decrease) occurs using dist-EF-SGDM for VGG-16, since VGG-16 has larger model size (528M).

[ResNet50_v2]
![image](https://user-images.githubusercontent.com/25879526/86327486-02f51380-bc76-11ea-8919-a66dcbc44862.png)

[VGG-16]
![image](https://user-images.githubusercontent.com/25879526/86327498-05576d80-bc76-11ea-95c6-b9f285193bb9.png)



#### Scaling Efficiency

We also measure scaling efficiency when the number of nodes varies from 1 to 8. We follow the same setup as in the above experiment. The figure shows that gradient compression improves the scaling efficiency. The efficiency gain in gradient compression is much higher for VGG-16 than ResNet-50_v2, since ResNet50_v2 has smaller communication overhead.

[ResNet50_v2]
![image](https://user-images.githubusercontent.com/25879526/86327513-0a1c2180-bc76-11ea-88a8-292f09d434b7.png)

[VGG-16]
![image](https://user-images.githubusercontent.com/25879526/86327520-0be5e500-bc76-11ea-9711-c5618923b956.png)

___
The above two sub-experiments were conducted 2 months ago. There have been large updates since then. So the results are a little outdated. They are just for reference. 

#### End-to-End Training

Finally, we train ResNet50_v2 and VGG-16 end-to-end to measure total reduction in training time. For such large batch training, warmup and linear scaling learning rate 
 are used to avoid generalization gap. We set the number of warmup epochs to 5. We also leverage cosine annealing strategy for learning rate decay. For ResNet50_v2 we use 8 AWS EC2 P3.16xlarge instances while for VGG-16, we use 4 AWS EC2 P3.16xlarge. 

[ResNet50_v2]
![image](https://user-images.githubusercontent.com/25879526/86327533-10120280-bc76-11ea-99ef-5c9e4c17e1bc.png)
![image](https://user-images.githubusercontent.com/25879526/86327537-11dbc600-bc76-11ea-84e6-bef6b88296b0.png)

As the figure shows, we reduce the trianing time by 8.0% without accuracy loss for ResNet50_v2.

|  6c44049fd49e532781af96add6a02a0427e6a1a8  | VAl ACC | TIME(h) |
| -- | -- | -- |
|sgdm| 0.76914465625| 2.6505945833029516|
|dist-ef-sgdm| 0.7632242968749999|2.4378090010373263 |

[VGG-16]
![image](https://user-images.githubusercontent.com/25879526/86327546-143e2000-bc76-11ea-8969-30c037f7022c.png)
![image](https://user-images.githubusercontent.com/25879526/86327556-16a07a00-bc76-11ea-943c-761b1b4dafbd.png)

The above figure shows that our implementation of dist-EF-SGDM reduces the training time for 100 epochs by 39.04% compared to the full-precision SGDM. We note that there is a small gap in accuracy between dist-EF-SGDM and SGDM. We will investigate this problem in the future.



## TODO

- [x] support inter-node compression
- [x] support intra-node for MXNet
- [x] support onebit compressor
- [x] support error-feedback
- [x] support momentum
- [ ] support other compressors
- [ ] support PyTorch and Tensorflow

## Precautions

1. To run successfully,  `ps-lite` should change one LOC. see the PR here. https://github.com/dmlc/ps-lite/pull/168
2. We only support Gluon for MXNet now. Raw MXNet's API does not support it.
3. Since gradient compression also has some overhead, this is a trade-off. It is only suitable for some cases, e.g. slow network or large models. In other cases, gradient compression will even harm performance.
4. Momentum here is the same as the framework's momentum. Why do we have to implement momentum again? This is because for some algorithms like [dist-EF-SGDM](https://papers.nips.cc/paper/9321-communication-efficient-distributed-blockwise-momentum-sgd-with-error-feedback.pdf) , momentum should be added first but many frameworks like MXNet exchange gradient first and then add the momentum. So we have to implement momentum inside BytePS. When inside momentum is used, outside momentum should be disabled (set \mu = 0) in the users' scripts.
5. FP16 is not supported now. 