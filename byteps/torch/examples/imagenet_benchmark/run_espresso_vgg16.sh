#!/bin/bash

path="`dirname $0`"
set -x

# set DMLC_PS_ROOT_URI with the IP address of the root GPU machine and ifname with the NIC name
ifname="eth0"
export DMLC_PS_ROOT_URI="10.188.181.138"

compress_ratio=0.01
gpus=0,1,2,3,4,5,6,7
export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-0}
export DMLC_INTERFACE=${ifname}
export NCCL_IB_DISABLE=1 
export NCCL_IB_GID_INDEX=3 
export NCCL_IB_HCA=mlx5_0 
export NCCL_SOCKET_IFNAME=${ifname}
export DMLC_NUM_WORKER=$1
export DMLC_NUM_SERVER=$DMLC_NUM_WORKER
export DMLC_NODE_HOST="$(/sbin/ip -o -4 addr list ${ifname} | awk '{print $4}' | cut -d/ -f1)"
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-12213}
export NVIDIA_VISIBLE_DEVICES=${gpus}
export BYTEPS_FORCE_DISTRIBUTED=0
export OMP_NUM_THREADS=4
export TEST_TYPE=${TEST_TYPE:=torch}
export NCCL_DEBUG=VERSION
# Ensure the NCCL_BUFFSIZE is larger than the message size of the compressed tensors 
export NCCL_BUFFSIZE=16777216
export DMLC_WORKER_ID=$2

IFS=', ' read -ra a <<< $gpus; 
gpus_per_node=${#a[@]}
declare -p a;

DISTRIBUTED_ARGS="--nproc_per_node ${gpus_per_node} --nnodes ${DMLC_NUM_WORKER} --node_rank ${DMLC_WORKER_ID} --master_addr ${DMLC_PS_ROOT_URI} --master_port 12345"

export BYTEPS_PARTITION_BYTES=4096000
pkill -9 python3
export NCCL_P2P_DISABLE=1

for model in "vgg16"
do
  for compressor in "randomk"
  do
    pkill -9 python3
    scheduler_file="../../mergeComp/scheduler/${model}/pcie_randomk_two_cpu"
    export BYTEPS_INTER_COMPRESSOR=${compressor}
    BENCHMARK_ARGS="--compress --compressor ${compressor} --memory topk --comm espresso --compress-ratio ${compress_ratio} --scheduler-file ${scheduler_file} --scheduler-type -1"
    $GDB python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/main.py --model ${model} --epochs 5 --batch-size 32 --speed_test $BENCHMARK_ARGS
    sleep 5
  done
done