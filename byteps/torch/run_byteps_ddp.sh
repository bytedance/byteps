#!/bin/bash

path="`dirname $0`"
set -x

compressor="randomk"
compress_ratio=0.01
communicator="allgather_twolayer"
gpus=0,1,2,3,4,5,6,7
export DMLC_ENABLE_RDMA=${DMLC_ENABLE_RDMA:-0}
export DMLC_INTERFACE="eth0"
export NCCL_IB_DISABLE=1 
export NCCL_IB_GID_INDEX=3 
export NCCL_IB_HCA=mlx5_0 
export NCCL_SOCKET_IFNAME=eth0
export DMLC_NUM_WORKER=$1
export DMLC_NUM_SERVER=$DMLC_NUM_WORKER
export DMLC_PS_ROOT_URI="$(host ${ARNOLD_WORKER_0_HOST} | head -1 | awk -F' ' '{print $NF}')"
export DMLC_NODE_HOST="$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)"
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-12213}
export NVIDIA_VISIBLE_DEVICES=${gpus}
export BYTEPS_FORCE_DISTRIBUTED=0
export BYTEPS_COMPRESSOR_ERROR_FEEDBACK="test"
export BYTEPS_INTER_COMPRESSOR=${compressor}
export OMP_NUM_THREADS=4
# export NCCL_P2P_DISABLE=1
export BYTEPS_MIN_COMPRESS_BYTES=1
#export BYTEPS_PARTITION_BYTES=4096000
#export BYTEPS_SERVER_ENGINE_THREAD=4
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-DEBUG}
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-TRACE}
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-INFO}
export PS_VERBOSE=${PS_VERBOSE:-0}
export TEST_TYPE=${TEST_TYPE:=torch}
# export NCCL_DEBUG=DEBUG
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

model='vgg16'
scheduler_file=$path/mergeComp/scheduler/${model}"_cpu_pcie"
DISTRIBUTED_ARGS="--nproc_per_node ${gpus_per_node} --nnodes ${DMLC_NUM_WORKER} --node_rank ${DMLC_WORKER_ID} --master_addr ${DMLC_PS_ROOT_URI} --master_port 12345"

export BYTEPS_PARTITION_BYTES=4096000

# $GDB python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/benchmark_byteps.py --model ${model} --num-iters 1000000
# $GDB python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/benchmark_byteps.py --model ${model} --num-iters 10 --fp16
# $GDB python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/test_ddp.py

for compressor in "randomk" "dgc" 
do
    for type in 5 #0 1 2 3 4 5 10
    do
        pkill -9 python3
        export BYTEPS_INTER_COMPRESSOR=${compressor}
        BENCHMARK_ARGS="--compress --compressor ${compressor} --memory topk --comm ${communicator} --compress-ratio ${compress_ratio} --scheduler-file ${scheduler_file} --scheduler-type ${type}"
        $GDB python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/benchmark_byteps.py --model ${model} --num-iters 10 $BENCHMARK_ARGS
        # python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/test_torch.py
        sleep 2
    done
done

for compressor in "efsignsgd" "onebit"
do
    pkill -9 python3
    export BYTEPS_INTER_COMPRESSOR=${compressor}
    BENCHMARK_ARGS="--compress --compressor ${compressor} --memory efsignsgd --comm ${communicator} --compress-ratio ${compress_ratio} --scheduler-file ${scheduler_file} --scheduler-type 0"
    $GDB python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/benchmark_byteps.py --model ${model} --num-iters 10 $BENCHMARK_ARGS
    # python3 -m torch.distributed.launch $DISTRIBUTED_ARGS $path/test_torch.py
    sleep 1
done