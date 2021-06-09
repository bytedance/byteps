#!/bin/bash

path="`dirname $0`"
set -x
ip=$(ip a show eth0 | awk '/inet / {print $2}' |  awk -F'[/]' '{print $1}')
RANK=$2
export PATH=~/.local/bin:$PATH
export BYTEPS_OMP_THREAD_PER_GPU=0
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$UCX_HOME/lib:$LD_LIBRARY_PATH
export DMLC_ENABLE_UCX=${DMLC_ENABLE_UCX:=1}
export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-2}
export DMLC_NUM_SERVER=$DMLC_NUM_WORKER
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-$ip}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-22130}
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$DMLC_PS_ROOT_URI}
export BYTEPS_ALLTOALL_SESSION_SIZE=2
export BYTEPS_LOCAL_SIZE=${BYTEPS_LOCAL_SIZE:-2}
export BYTEPS_LOCAL_RANK=$((RANK % BYTEPS_LOCAL_SIZE))
export UCX_RDMA_CM_SOURCE_ADDRESS=$DMLC_NODE_HOST
export BYTEPS_NUMA_ID=$(($2 / 4))

export UCX_RNDV_THRESH=8k
export UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-2}
export UCX_USE_MT_MUTEX=y
export UCX_SOCKADDR_CM_ENABLE=y
export UCX_IB_TRAFFIC_CLASS=236

export DMLC_NUM_CPU_DEV=0
export DMLC_NUM_GPU_DEV=1
export CUDA_VISIBLE_DEVICES=$(($BYTEPS_LOCAL_RANK))

export BYTEPS_SERVER_DIRECT_RESPONSE=${BYTEPS_SERVER_DIRECT_RESPONSE:-0}
export BYTEPS_WORKER_LOCAL_ROOT=0
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_PARTITION_BYTES=${BYTEPS_PARTITION_BYTES:-4096000}
export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-info}
export PS_VERBOSE=${PS_VERBOSE:-0}
export TEST_TYPE=${TEST_TYPE:=tensorflow}
export BYTEPS_TELEMETRY_ON=0
export BYTEPS_UCX_SHORT_THRESH=0
# Note: updated BYTEPS_SOCKET_PATH for cpu allreduce
export BYTEPS_PHY_NODE_ID=$((RANK / BYTEPS_LOCAL_SIZE))
export BYTEPS_SOCKET_PATH=/tmp/node-${BYTEPS_PHY_NODE_ID}
mkdir -p $BYTEPS_SOCKET_PATH

if [ $1 == "scheduler" ]; then
  echo "Launch scheduler"
  CUDA_VISIBLE_DEVICES=0 DMLC_ROLE=scheduler python3 -c 'import byteps.server'
  exit
fi

export DMLC_WORKER_ID=$2
if [ $1 == "server" ]; then
  echo "Launch server"
  DMLC_ROLE=server python3 -c 'import byteps.server'
  exit
fi

if [ "$#" -ne 3 ]; then
  echo "Usage: bash run_byteps_test_byteccl.sh role rank test_script"
  exit
fi
BIN="$3"
# export GDB=" gdb -ex run --args "
export GDB=" "

if [ $1 == "worker" ] || [ $1 == "joint" ]; then
  export DMLC_ROLE=$1
  if [ "$TEST_TYPE" == "tensorflow" ]; then
    echo "TEST TENSORFLOW ..."
    $GDB python3 $path/$BIN --rank $2
  elif [ "$TEST_TYPE" == "torch" ]; then
    echo "TEST TORCH ..."
    python3 $path/$BIN --rank $2
  else
    echo "Error: unsupported $TEST_TYPE"
    exit 1
  fi
fi
