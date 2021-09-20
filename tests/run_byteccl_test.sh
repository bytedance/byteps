#!/bin/bash

path="`dirname $0`"
set -x
ip=$(ip a show eth0 | awk '/inet / {print $2}' |  awk -F'[/]' '{print $1}')
RANK=$2
export PATH=~/.local/bin:$PATH
export BYTEPS_OMP_THREAD_PER_GPU=0
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$UCX_HOME/lib:$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export DMLC_ENABLE_UCX=${DMLC_ENABLE_UCX:=1}
export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-2}
export DMLC_NUM_SERVER=$DMLC_NUM_WORKER
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-$ip}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-22130}
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$DMLC_PS_ROOT_URI}
export BYTEPS_LOCAL_SIZE=${BYTEPS_LOCAL_SIZE:-2}
export BYTEPS_LOCAL_RANK=$((RANK % BYTEPS_LOCAL_SIZE))
export UCX_RDMA_CM_SOURCE_ADDRESS=$DMLC_NODE_HOST
export UCX_TLS=${UCX_TLS:=rc_x,cuda_copy,cuda_ipc,tcp}
export BYTEPS_PIN_MEMORY=${BYTEPS_PIN_MEMORY:-1}

export UCX_RNDV_THRESH=8k
export UCX_IB_TRAFFIC_CLASS=236
export DMLC_NUM_CPU_DEV=${DMLC_NUM_CPU_DEV:=0}
export DMLC_NUM_GPU_DEV=${DMLC_NUM_GPU_DEV:=1}

export BYTEPS_SERVER_DIRECT_RESPONSE=${BYTEPS_SERVER_DIRECT_RESPONSE:-0}
export BYTEPS_WORKER_LOCAL_ROOT=-1
export BYTEPS_FORCE_DISTRIBUTED=${BYTEPS_FORCE_DISTRIBUTED:=1}
export BYTEPS_PARTITION_BYTES=${BYTEPS_PARTITION_BYTES:-4096000}
export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-info}
export PS_VERBOSE=${PS_VERBOSE:-0}
export TEST_TYPE=${TEST_TYPE:=tensorflow}
export BYTEPS_TELEMETRY_ON=1
export BYTEPS_TELEMETRY_INTERVAL=1
export BYTEPS_UCX_SHORT_THRESH=0
# Note: updated BYTEPS_SOCKET_PATH for cpu allreduce
export BYTEPS_PHY_NODE_ID=$((RANK / BYTEPS_LOCAL_SIZE))
export BYTEPS_SOCKET_PATH=/tmp/node-${BYTEPS_PHY_NODE_ID}
export BYTEPS_UUID=${BYTEPS_UUID+$BYTEPS_PHY_NODE_ID}
mkdir -p $BYTEPS_SOCKET_PATH

if [ $1 == "scheduler" ]; then
  echo "Launch scheduler"
  CUDA_VISIBLE_DEVICES=0 DMLC_ROLE=scheduler python3 -c 'import byteps.server'
  exit $?
fi

export BYTEPS_NUMA_ID=$(($2 / 4))
export DMLC_WORKER_ID=$BYTEPS_PHY_NODE_ID

if [ $1 == "server" ]; then
  echo "Launch server"
  DMLC_ROLE=server python3 -c 'import byteps.server'
  exit $?
fi

if [ "$#" -ne 3 ]; then
  echo "Usage: bash run_byteps_test_byteccl.sh role rank test_script"
  exit
fi
BIN="$3"


if [ "$RANK" == "0" ]; then
    export NSYS="nsys profile -t cuda,nvtx,cublas --stats=true -o ./trace.%p.prof --export sqlite --kill none -d 30"
fi
export NSYS=" "
export GDB="gdb -ex run  -ex bt -batch --args"
export GDB=" "

if [ $1 == "joint" ]; then
  export DMLC_WORKER_ID=$2
fi

if [ $1 == "worker" ] || [ $1 == "joint" ]; then
  export DMLC_ROLE=$1
  if [ "$TEST_TYPE" == "tensorflow" ]; then
    echo "TEST TENSORFLOW ..."
    export CUDA_VISIBLE_DEVICES=$BYTEPS_LOCAL_RANK
    $GDB python3 $path/$BIN
    ret_code=$?
  elif [ "$TEST_TYPE" == "torch" ]; then
    echo "TEST TORCH ..."
    $GDB python3 $path/$BIN
    ret_code=$?
  else
    echo "Error: unsupported $TEST_TYPE"
    exit 1
  fi
fi
echo "Done $TEST_TYPE test"
exit $ret_code
