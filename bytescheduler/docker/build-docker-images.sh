s#!/bin/bash

set -e
set -x

#docker build -t bsc-mxnet-ps:$(date +%Y%m%d-%H%M%S) --no-cache -f mxnet_ps.Dockerfile .

docker build -t bsc-mxnet-ps-ssh:$(date +%Y%m%d-%H%M%S) --no-cache -f mxnet_ps_ssh.Dockerfile .

#docker build -t bsc-mxnet-horovod:$(date +%Y%m%d-%H%M%S) --no-cache -f mxnet_horovod.Dockerfile .

#docker build -t bsc-pytorch-horovod:$(date +%Y%m%d-%H%M%S) --no-cache -f pytorch_horovod.Dockerfile .

#docker build -t bsc-pytorch-horovod-ssh:$(date +%Y%m%d-%H%M%S) --no-cache -f pytorch_horovod_ssh.Dockerfile .

#docker build -t test --no-cache -f dockerfile .