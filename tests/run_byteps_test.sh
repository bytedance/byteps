#!/bin/bash

if [ "$TEST_TYPE" == "mxnet" ]; then
  echo "TEST MXNET ..."
  python /opt/tiger/byteps/tests/test_mxnet.py $@
elif [ "$TEST_TYPE" == "keras" ]; then
  echo "TEST KERAS ..."
  python /opt/tiger/byteps/tests/test_tensorflow_keras.py $@
else
  echo "Error: unsupported $TEST_TYPE"
  exit 1
fi
