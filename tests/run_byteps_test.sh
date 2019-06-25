#!/bin/bash

path="`dirname $0`"

if [ "$TEST_TYPE" == "mxnet" ]; then
  echo "TEST MXNET ..."
  python $path/test_mxnet.py $@
elif [ "$TEST_TYPE" == "keras" ]; then
  echo "TEST KERAS ..."
  python $path/test_tensorflow_keras.py $@
else
  echo "Error: unsupported $TEST_TYPE"
  exit 1
fi
