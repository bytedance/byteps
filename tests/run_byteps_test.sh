#!/bin/bash

path="$(dirname $0)"

#export PATH=~/.local/bin:$PATH
export LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH
export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-2}
export DMLC_NUM_SERVER=$DMLC_NUM_WORKER
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-10.188.137.23}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-22210}
export DMLC_NODE_HOST=${DMLC_NODE_HOST:-$DMLC_PS_ROOT_URI}
export BYTEPS_LOCAL_RANK=0
export BYTEPS_LOCAL_SIZE=1
export NVIDIA_VISIBLE_DEVICES=0
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_COMPRESSOR=signsgd
export BYTEPS_PARTITION_BYTES=4096000
#export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-TRACE}
export BYTEPS_LOG_LEVEL=${BYTEPS_LOG_LEVEL:-DEBUG}
export PS_VERBOSE=${PS_VERBOSE:-0}
export TEST_TYPE=${TEST_TYPE:=torch}

function cleanup() {
  rm -rf lr.s
}

trap cleanup EXIT

pkill bpslaunch
pkill python3

if [ $1 == "scheduler" ]; then
  echo "Launch scheduler"
  DMLC_ROLE=scheduler python3 -c 'import byteps.server'
  exit
fi


export DMLC_WORKER_ID=$2
if [ $1 == "server" ]; then
  echo "Launch server"
  DMLC_ROLE=server python3 -c 'import byteps.server'
  exit
fi

#export GDB=" gdb -ex run --args "
export GDB=" "

if [ $1 == "worker" ] || [ $1 == "joint" ]; then
  export DMLC_ROLE=$1
  if [ "$TEST_TYPE" == "torch" ]; then
    echo "TEST TORCH ..."
    $GDB python3 $path/benchmark_byteps.py
  else
    echo "Error: unsupported $TEST_TYPE"
    exit 1
  fi
fi