#batch_size = 80
mkdir baseline_results

#!/bin/bash

path="`dirname $0`"
set -x

compress_ratio=0.001
gpus=0,1,2,3,4,5,6,7
export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-0}
export DMLC_INTERFACE="eth0"
export NCCL_IB_DISABLE=1 
export NCCL_IB_GID_INDEX=3 
export NCCL_IB_HCA=mlx5_0 
export NCCL_SOCKET_IFNAME=eth0
export DMLC_NUM_WORKER=$1
export DMLC_NUM_SERVER=$DMLC_NUM_WORKER
# export DMLC_PS_ROOT_URI="$(host ${ARNOLD_WORKER_0_HOST} | head -1 | awk -F' ' '{print $NF}')"
export DMLC_PS_ROOT_URI="10.188.180.85"
export DMLC_NODE_HOST="$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)"
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-12213}
export NVIDIA_VISIBLE_DEVICES=${gpus}
export BYTEPS_FORCE_DISTRIBUTED=0
# export BYTEPS_COMPRESSOR_ERROR_FEEDBACK="test"
export BYTEPS_COMPRESSOR_k=${compress_ratio}
export OMP_NUM_THREADS=4
# export NCCL_P2P_DISABLE=1
#export BYTEPS_PARTITION_BYTES=4096000
#export BYTEPS_SERVER_ENGINE_THREAD=4
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-DEBUG}
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-TRACE}
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-INFO}
export PS_VERBOSE=${PS_VERBOSE:-0}
export TEST_TYPE=${TEST_TYPE:=torch}
export NCCL_DEBUG=DEBUG
# Ensure the NCCL_BUFFSIZE is larger than the message size of the compressed tensors 
export NCCL_BUFFSIZE=16777216
export BYTEPS_TENSOR_INFO=1
#export BYTEPS_ENABLE_GDB=1
# export BYTEPS_TRACE_ON=1
# export BYTEPS_TRACE_END_STEP=40
# export BYTEPS_TRACE_START_STEP=20
# export BYTEPS_TRACE_DIR=./traces
#export GDB=" gdb -ex run --args "
#export GDB=" "
export DMLC_WORKER_ID=$2

IFS=', ' read -ra a <<< $gpus; 
gpus_per_node=${#a[@]}
declare -p a;

model='LSTM'
DISTRIBUTED_ARGS="--nproc_per_node ${gpus_per_node} --nnodes ${DMLC_NUM_WORKER} --node_rank ${DMLC_WORKER_ID} --master_addr ${DMLC_PS_ROOT_URI} --master_port 12345"

export BYTEPS_PARTITION_BYTES=4096000
pkill -9 python3

dirname="bytecomp_results"
mkdir $dirname

for pcie in 1
do 
  export NCCL_P2P_DISABLE=${pcie}
  for communicator in "bytecomp"
  do
      for compressor in "efsignsgd" "onebit"
      do
        pkill -9 python3
        export BYTEPS_INTER_COMPRESSOR=${compressor}
        if [ ${pcie} -eq 0 ]; then
          scheduler_file="../../mergeComp/scheduler/lstm/nvlink_${compressor}_cpu"
        else
          scheduler_file="../../mergeComp/scheduler/lstm/pcie_onebit_two_cpu"
        fi        
        BENCHMARK_ARGS="--compress --compressor ${compressor} --memory efsignsgd --comm ${communicator} --compress-ratio ${compress_ratio} --scheduler-file ${scheduler_file} --scheduler-type -1"
        python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/main.py --model ${model} --epochs 2 $BENCHMARK_ARGS | tee -a ${dirname}/pcie_${pcie}_${compressor}_${communicator}_${DMLC_NUM_WORKER}
        sleep 5
      done
  done
done