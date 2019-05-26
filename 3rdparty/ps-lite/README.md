<img src="http://parameterserver.org/images/parameterserver.png"  width=400 />

[![Build Status](https://travis-ci.org/dmlc/ps-lite.svg?branch=master)](https://travis-ci.org/dmlc/ps-lite)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

A light and efficient implementation of the parameter server
framework. It provides clean yet powerful APIs. For example, a worker node can
communicate with the server nodes by
- `Push(keys, values)`: push a list of (key, value) pairs to the server nodes
- `Pull(keys)`: pull the values from servers for a list of keys
- `Wait`: wait untill a push or pull finished.

A simple example:

```c++
  std::vector<uint64_t> key = {1, 3, 5};
  std::vector<float> val = {1, 1, 1};
  std::vector<float> recv_val;
  ps::KVWorker<float> w;
  w.Wait(w.Push(key, val));
  w.Wait(w.Pull(key, &recv_val));
```

More features:

- Flexible and high-performance communication: zero-copy push/pull, supporting
  dynamic length values, user-defined filters for communication compression
- Server-side programming: supporting user-defined handles on server nodes

### Build

`ps-lite` requires a C++11 compiler such as `g++ >= 4.8`. On Ubuntu >= 13.10, we
can install it by
```
sudo apt-get update && sudo apt-get install -y build-essential git
```
Instructions for
[older Ubuntu](http://ubuntuhandbook.org/index.php/2013/08/install-gcc-4-8-via-ppa-in-ubuntu-12-04-13-04/),
[Centos](http://linux.web.cern.ch/linux/devtoolset/),
and
[Mac Os X](http://hpc.sourceforge.net/).

Then clone and build

```bash
git clone https://github.com/dmlc/ps-lite
cd ps-lite && make -j4
```

### Build with RDMA support

You can add `USE_RDMA=1` to enable RDMA support.

```bash
make -j $(nproc) USE_RDMA=1
```

### Enable new features for higher performance

To avoid head-of-line blocking, use `USE_MULTI_THREAD_FOR_RECEIVING=1` to enable multi-threading 
in the receiving functions of `customer.cc` 
(this flag must be set explicitly to enable multi-threading).
 You can also set `NUM_RECEIVE_THREAD` to change the number of threads of the thread pool 
(default is 2).


### How to use

`ps-lite` provides asynchronous communication for other projects: 
  - Distributed deep neural networks:
    [MXNet](https://github.com/dmlc/mxnet),
    [CXXNET](https://github.com/dmlc/cxxnet) and
    [Minverva](https://github.com/minerva-developers/minerva)
  - Distributed high dimensional inference, such as sparse logistic regression,
    factorization machines:
    [DiFacto](https://github.com/dmlc/difacto)
    [Wormhole](https://github.com/dmlc/wormhole)

### History

We started to work on the parameter server framework since 2010.

1. The first generation was
designed and optimized for specific algorithms, such as logistic regression and
LDA, to serve the sheer size industrial machine learning tasks (hundreds billions of
examples and features with 10-100TB data size) .

2. Later we tried to build a open-source general purpose framework for machine learning
algorithms. The project is available at [dmlc/parameter_server](https://github.com/dmlc/parameter_server).

3. Given the growing demands from other projects, we created `ps-lite`, which provides a clean data communication API and a
lightweight implementation. The implementation is based on `dmlc/parameter_server`, but we refactored the job launchers, file I/O and machine
learning algorithms codes into different projects such as `dmlc-core` and
`wormhole`.

4. From the experience we learned during developing
   [dmlc/mxnet](https://github.com/dmlc/mxnet), we further refactored the API and implementation from [v1](https://github.com/dmlc/ps-lite/releases/tag/v1). The main
   changes include
   - less library dependencies
   - more flexible user-defined callbacks, which facilitate other language
   bindings
   - let the users, such as the dependency
     engine of mxnet, manage the data consistency

### Research papers
  1. Mu Li, Dave Andersen, Alex Smola, Junwoo Park, Amr Ahmed, Vanja Josifovski,
     James Long, Eugene Shekita, Bor-Yiing
     Su. [Scaling Distributed Machine Learning with the Parameter Server](http://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf). In
     Operating Systems Design and Implementation (OSDI), 2014
  2. Mu Li, Dave Andersen, Alex Smola, and Kai
     Yu. [Communication Efficient Distributed Machine Learning with the Parameter Server](http://www.cs.cmu.edu/~muli/file/parameter_server_nips14.pdf). In
     Neural Information Processing Systems (NIPS), 2014
