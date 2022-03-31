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
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_COMPRESSOR_ERROR_FEEDBACK=""
export BYTEPS_INTER_COMPRESSOR=${compressor}
export BYTEPS_COMPRESSOR_K=0.01
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

model='bert'
scheduler_file=$path/mergeComp/scheduler/${model}"_cpu_pcie"
DISTRIBUTED_ARGS="--nproc_per_node ${gpus_per_node} --nnodes ${DMLC_NUM_WORKER} --node_rank ${DMLC_WORKER_ID} --master_addr ${DMLC_PS_ROOT_URI} --master_port 12345"

epochs="2.0"
# init_checkpoint="./dataset/checkpoint/bert_large_qa.pt"
init_checkpoint="./dataset/checkpoint/bert_base_qa.pt"
learning_rate="3e-5"
precision="fp32"
seed="1"
squad_dir="./dataset/squad/v1.1"
vocab_file="./dataset/vocab.txt"
OUT_DIR="."
mode="train"
# CONFIG_FILE="./dataset/checkpoint/bert_config.json"
CONFIG_FILE="./dataset/checkpoint/bert_base_config.json"
# max_steps is used for benchmark by terminating the training with the steps of max_steps
max_steps="100"
batch_size="8"

BERT_ARGS+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  BERT_ARGS+="--do_train "
  BERT_ARGS+="--train_file=$squad_dir/train-v1.1.json "
  BERT_ARGS+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  BERT_ARGS+="--do_predict "
  BERT_ARGS+="--predict_file=$squad_dir/dev-v1.1.json "
  BERT_ARGS+="--predict_batch_size=$batch_size "
elif [ "$mode" = "prediction" ] ; then
  BERT_ARGS+="--do_predict "
  BERT_ARGS+="--predict_file=$squad_dir/dev-v1.1.json "
  BERT_ARGS+="--predict_batch_size=$batch_size "
else
  BERT_ARGS+=" --do_train "
  BERT_ARGS+=" --train_file=$squad_dir/train-v1.1.json "
  BERT_ARGS+=" --train_batch_size=$batch_size "
  BERT_ARGS+="--do_predict "
  BERT_ARGS+="--predict_file=$squad_dir/dev-v1.1.json "
  BERT_ARGS+="--predict_batch_size=$batch_size "
fi
BERT_ARGS+=" --do_lower_case "
# CMD+=" --old "
# CMD+=" --loss_scale=128 "
BERT_ARGS+=" --bert_model=bert-large-uncased "
BERT_ARGS+=" --learning_rate=$learning_rate "
BERT_ARGS+=" --seed=$seed "
BERT_ARGS+=" --num_train_epochs=$epochs "
BERT_ARGS+=" --max_seq_length=384 "
BERT_ARGS+=" --doc_stride=128 "
BERT_ARGS+=" --output_dir=$OUT_DIR "
BERT_ARGS+=" --vocab_file=$vocab_file "
BERT_ARGS+=" --config_file=$CONFIG_FILE "
BERT_ARGS+=" --max_steps=$max_steps "
BERT_ARGS+=" --examples_limit=10000 "
BERT_ARGS+=" --log_freq=10 "

for pcie in 0 1
do 
  export NCCL_P2P_DISABLE=${pcie}
  # run FP32 with BytePS
  python3 -m torch.distributed.launch $DISTRIBUTED_ARGS run_squad.py ${BERT_ARGS} 
done
  
for pcie in 0 1
do 

  for communicator in "allgather" "hitopkcomm" "hipress"
  do
    for compressor in "randomk" "dgc"
    do
      pkill -9 python3
      BENCHMARK_ARGS="--compress --compressor ${compressor} --memory topk --comm ${communicator} --compress-ratio ${compress_ratio} --scheduler-file none --scheduler-type 0"
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS run_squad.py ${BERT_ARGS} $BENCHMARK_ARGS | tee -a bytecomp_logs/pcie_${pcie}_${compressor}_${DMLC_NUM_WORKER}
      sleep 5
    done

    for compressor in "efsignsgd" "onebit"
    do
      pkill -9 python3
      export BYTEPS_INTER_COMPRESSOR=${compressor}
      BENCHMARK_ARGS="--compress --compressor ${compressor} --memory efsignsgd --comm ${communicator} --compress-ratio ${compress_ratio} --scheduler-file none --scheduler-type 0"
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS run_squad.py ${BERT_ARGS} $BENCHMARK_ARGS | tee -a bytecomp_logs/pcie_${pcie}_${compressor}_${DMLC_NUM_WORKER}
      sleep 5
    done
  done

done