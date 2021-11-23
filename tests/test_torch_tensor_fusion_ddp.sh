#!/bin/bash

set -ex

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FILE=$THIS_DIR/torch/test_torch_tensor_fusion_ddp.py

NVIDIA_VISIBLE_DEVICES=0,2

export eth2_ip=$(hostname -I | cut -d' ' -f1)
eval " export MASTER_ADDR=${MASTER_ADDR:-$eth2_ip}"
export MASTER_PORT=23456
export ML_PLATFORM_WORKER_GPU=$(echo $NVIDIA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export ML_PLATFORM_WORKER_NUM=${ML_PLATFORM_WORKER_NUM:-1}
export NODE_RANK=${NODE_RANK:-0}

export LOCAL_WORLD_SIZE=${ML_PLATFORM_WORKER_GPU}
export GROUP_RANK=$NODE_RANK

export DMLC_NODE_HOST=$eth2_ip

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BYTEPS_BUCKET_PROBE_STEPS=1000
export BYTEPS_LOG_LEVEL=INFO
run_test () {
  if [[ "$CPU_ONLY" == "1" ]]; then
    python3 -m torch.distributed.launch --use_env \
             --nproc_per_node=${ML_PLATFORM_WORKER_GPU} \
             --nnodes=${ML_PLATFORM_WORKER_NUM} \
             --node_rank=${NODE_RANK} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             "$@" --cpu-only
  else
    python3 -m torch.distributed.launch --use_env \
             --nproc_per_node=${ML_PLATFORM_WORKER_GPU} \
             --nnodes=${ML_PLATFORM_WORKER_NUM} \
             --node_rank=${NODE_RANK} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             "$@"
  fi

  if [[ $? != "0" ]]; then
    exit $?
  fi
}

# SGD
run_test $FILE --optimizer SGD --lr 0.01 --momentum 0.9 --weight_decay 4e-5 --num-iters 1000
run_test $FILE --optimizer SGD --lr 0.01 --momentum 0.9 --weight_decay 4e-5 --num-iters 1000 --compare-apex

# Adam
run_test $FILE --optimizer Adam
run_test $FILE --optimizer Adam --beta1 0.999
run_test $FILE --optimizer Adam --beta1 0.999 --beta2 0.8
run_test $FILE --optimizer Adam --eps 5e-8

# Adagrad
run_test $FILE --optimizer Adagrad
run_test $FILE --optimizer Adagrad --weight_decay 2e-4
run_test $FILE --optimizer Adagrad --eps 1e-7

echo 'success'
