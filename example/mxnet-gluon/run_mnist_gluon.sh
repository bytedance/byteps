#!/bin/bash

export NVIDIA_VISIBLE_DEVICES=0,1
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=1
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000

path="`dirname $0`"
echo $path

python $path/../../launcher/launch.py \
	python3 $path/train_mnist_byteps.py