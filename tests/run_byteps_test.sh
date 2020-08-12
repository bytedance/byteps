#!/bin/bash

path="$(dirname $0)"

export PATH=~/anaconda3/envs/mxnet_p36/bin:$PATH
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234

function cleanup() {
  rm -rf lr.s
}

trap cleanup EXIT

pkill bpslaunch
pkill python3

echo "Launch scheduler"
export DMLC_ROLE=scheduler
bpslaunch &

echo "Launch server"
export DMLC_ROLE=server
bpslaunch &

export NVIDIA_VISIBLE_DEVICES=0
export DMLC_WORKER_ID=0
export DMLC_ROLE=worker
export BYTEPS_THREADPOOL_SIZE=4
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_LOG_LEVEL=WARNING

if [ "$TEST_TYPE" == "keras" ]; then
  echo "TEST KERAS ..."
  python $path/test_tensorflow_keras.py $@
else
  echo "Error: unsupported $TEST_TYPE"
  exit 1
fi
