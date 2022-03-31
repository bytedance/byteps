#!/bin/bash

path="`dirname $0`"
set -x

compress_ratio=0.01
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
export BYTEPS_FORCE_DISTRIBUTED=1
export OMP_NUM_THREADS=4
# export NCCL_P2P_DISABLE=1
#export BYTEPS_PARTITION_BYTES=4096000
#export BYTEPS_SERVER_ENGINE_THREAD=4
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-DEBUG}
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-TRACE}
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-INFO}
export PS_VERBOSE=${PS_VERBOSE:-0}
export TEST_TYPE=${TEST_TYPE:=torch}
# export NCCL_DEBUG=DEBUG
# Ensure the NCCL_BUFFSIZE is larger than the message size of the compressed tensors 
export NCCL_BUFFSIZE=16777216
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

model='gpt2'
DISTRIBUTED_ARGS="--nproc_per_node ${gpus_per_node} --nnodes ${DMLC_NUM_WORKER} --node_rank ${DMLC_WORKER_ID} --master_addr ${DMLC_PS_ROOT_URI} --master_port 12345"

export DATA_DIR=${DATA_DIR:-~/data}
export TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/wikitext-2-raw/wiki.train.raw}
export TEST_FILE=${TRAIN_FILE:-$DATA_DIR/wikitext-2-raw/wiki.test.raw}
export DISTRIBUTED_FRAMEWORK=${DISTRIBUTED_FRAMEWORK:-byteps}
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ~ 
cd $THIS_DIR/gpt-2/examples

GPT2_ARGS="--train_data_file=$TRAIN_FILE --output_dir=output --model_type=gpt2 --model_name_or_path=gpt2 --do_train --save_steps 1000000 --overwrite_output_dir --num_train_epochs 3 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 4"
# pcie=1


for pcie in 0 1
do 
    export NCCL_P2P_DISABLE=1

    for compressor in "randomk" "dgc" 
    do
        pkill -9 python3
        export BYTEPS_INTER_COMPRESSOR=${compressor}
        if [ ${pcie} -eq 0 ]; then
          scheduler_file="../../mergeComp/scheduler/${model}/nvlink_${compressor}_cpu"
        else
          scheduler_file="../../mergeComp/scheduler/${model}/pcie_${compressor}_two_cpu"
        fi
        BENCHMARK_ARGS="--compress --compressor ${compressor} --memory topk --comm bytecomp --compress-ratio ${compress_ratio} --scheduler-file ${scheduler_file} --scheduler-type -1"
        python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} run_lm_finetuning_bytecomp.py ${GPT2_ARGS} $BENCHMARK_ARGS | tee -a ../../bytecomp_logs/pcie_${pcie}_${compressor}_${DMLC_NUM_WORKER}
        sleep 5
    done

    for compressor in "onebit" "efsignsgd" 
    do
        pkill -9 python3
        export BYTEPS_INTER_COMPRESSOR=${compressor}
        if [ ${pcie} -eq 0 ]; then
          scheduler_file="../../mergeComp/scheduler/${model}/nvlink_${compressor}_cpu"
        else
          scheduler_file="../../mergeComp/scheduler/${model}/pcie_${compressor}_two_cpu"
        fi
        BENCHMARK_ARGS="--compress --compressor ${compressor} --memory efsignsgd --comm bytecomp --compress-ratio ${compress_ratio} --scheduler-file ${scheduler_file} --scheduler-type -1"
        python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} run_lm_finetuning_bytecomp.py ${GPT2_ARGS} $BENCHMARK_ARGS | tee -a ../../bytecomp_logs/pcie_${pcie}_${compressor}_${DMLC_NUM_WORKER}
        sleep 5
    done

done