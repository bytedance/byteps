#!/bin/bash

set -x
tmux clear-history; clear;
export BYTEPS_LOG_LEVEL=DEBUG
#export BYTEPS_LOG_LEVEL=TRACE
export CUDA_VISIBLE_DEVICES=0,1
export NVIDIA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
# export BYTEPS_FORCE_DISTRIBUTED=1

# export CUDA_VISIBLE_DEVICES=0
# export NVIDIA_VISIBLE_DEVICES=0
# export BYTEPS_LOCAL_RANK=0
# export BYTEPS_LOCAL_SIZE=2
# python3 tensorflow2_mnist.py  &
#
# export CUDA_VISIBLE_DEVICES=1
# export NVIDIA_VISIBLE_DEVICES=1
# export BYTEPS_LOCAL_RANK=1

# BYTEPSRUN gdb -ex run -ex bt -batch --args python3 tensorflow2_mnist.py 2>&1 | tee test.txt
BYTEPSRUN python3 tensorflow2_mnist.py 2>&1 | tee test.txt
