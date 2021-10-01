#!/bin/bash

set -ex

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FILE=$THIS_DIR/test_torch_tensor_fusion.py

export DMLC_WORKER_ID=0 DMLC_NUM_WORKER=1 DMLC_ROLE=worker NVIDIA_VISIBLE_DEVICES=0,1

# SGD
bpslaunch python3 $FILE --optimizer SGD --lr 0.01 --momentum 0.9 --weight_decay 4e-5 --num-iters 1000
if [[ $? != "0" ]]; then
  exit $?
fi
bpslaunch python3 $FILE --optimizer SGD --lr 0.01 --momentum 0.9 --weight_decay 4e-5 --num-iters 1000 --compare-apex
if [[ $? != "0" ]]; then
  exit $?
fi

# Adam
bpslaunch python3 $FILE --optimizer Adam
if [[ $? != "0" ]]; then
  exit $?
fi
bpslaunch python3 $FILE --optimizer Adam --beta1 0.999
if [[ $? != "0" ]]; then
  exit $?
fi
bpslaunch python3 $FILE --optimizer Adam --beta1 0.999 --beta2 0.8
if [[ $? != "0" ]]; then
  exit $?
fi
bpslaunch python3 $FILE --optimizer Adam --eps 5e-8
if [[ $? != "0" ]]; then
  exit $?
fi

# Adagrad
bpslaunch python3 $FILE --optimizer Adagrad
if [[ $? != "0" ]]; then
  exit $?
fi
bpslaunch python3 $FILE --optimizer Adagrad --weight_decay 2e-4
if [[ $? != "0" ]]; then
  exit $?
fi
bpslaunch python3 $FILE --optimizer Adagrad --eps 1e-7
if [[ $? != "0" ]]; then
  exit $?
fi

echo 'success'
