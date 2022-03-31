#!/bin/bash

path="`dirname $0`"
set -x

compressor="dgc"
compress_ratio=0.05
export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-0}
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=$DMLC_NUM_WORKER
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-10.188.137.204}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-22210}
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$DMLC_PS_ROOT_URI}
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_COMPRESSOR_ERROR_FEEDBACK=""
export BYTEPS_COMPRESSOR=${compressor}
# BYTEPS_INTRA_COMPRESSOR is to tell if intra-node compression is needed
export BYTEPS_INTRA_COMPRESSOR=${compressor}
export BYTEPS_COMPRESSOR_K=${compress_ratio}
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=0
export BYTEPS_MIN_COMPRESS_BYTES=1
export BYTEPS_PARTITION_BYTES=44096000
#export BYTEPS_PARTITION_BYTES=2048000
#export BYTEPS_PARTITION_BYTES=48192000
export BYTEPS_PUSH_THREADS=1
#export BYTEPS_SERVER_ENGINE_THREAD=4
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-DEBUG}
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-TRACE}
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-INFO}
export PS_VERBOSE=${PS_VERBOSE:-0}
export TEST_TYPE=${TEST_TYPE:=torch}
#export BYTEPS_ENABLE_GDB=1
# export BYTEPS_TRACE_ON=1
# export BYTEPS_TRACE_END_STEP=40
# export BYTEPS_TRACE_START_STEP=20
# export BYTEPS_TRACE_DIR=./traces

#export GDB=" gdb -ex run --args "
export GDB=" "

if [ $1 == "scheduler" ]; then
  echo "Launch scheduler"
  DMLC_ROLE=scheduler python3.7 -c 'import byteps.server'
  exit
fi

export DMLC_WORKER_ID=$2
if [ $1 == "server" ]; then
  echo "Launch server"
  DMLC_ROLE=server bpslaunch python3.7 -c 'import byteps.server'
  exit
fi

if [ $1 == "worker" ] || [ $1 == "joint" ]; then
  export DMLC_ROLE=$1
  echo "TEST TORCH ..."
  #bpslaunch python3.7 $path/benchmark_byteps.py --model resnet50 --num-iters 2 #--compress --compressor ${compressor} --memory none --comm byteps_compress --compress-ratio ${compress_ratio}--fusion-num $3
  #bpslaunch python3.7 $path/benchmark_byteps_cifar10.py --model resnet50 #--compress --compressor ${compressor} --memory none --comm byteps_compress --compress-ratio ${compress_ratio} --fusion-num $3
  bpslaunch python3.7 $path/train_imagenet_resnet50_byteps.py --model resnet50 --epochs 2 --compress --compressor ${compressor} --memory none --comm byteps_compress --compress-ratio ${compress_ratio} --fusion-num $3
fi