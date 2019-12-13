#!/bin/bash

path="`dirname $0`"

export NVIDIA_VISIBLE_DEVICES=0,1,2,3
export DMLC_NUM_WORKER=1
export DMLC_WORKER_ID=0
export DMLC_ROLE=worker

if [ "$TEST_TYPE" == "mxnet" ]; then
  echo "TEST MXNET ..."
  python3 /usr/local/byteps/launcher/launch.py python3 $path/test_mxnet.py $@
elif [ "$TEST_TYPE" == "keras" ]; then
  echo "TEST KERAS ..."
  python $path/test_tensorflow_keras.py $@
else
  echo "Error: unsupported $TEST_TYPE"
  exit 1
fi
