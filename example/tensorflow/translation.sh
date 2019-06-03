#!/bin/bash

if [ "$ARNOLD_ROLE" == "server" ]; then
  ARNOLD_ROLE="ps"
fi

echo $ARNOLD_ROLE, $ARNOLD_ID, $ARNOLD_SERVER_HOSTS, $ARNOLD_WORKER_HOSTS, $ARNOLD_OUTPUT

sed -i -e "s/per_process_gpu_memory_fraction=gpu_memory_fraction)/allow_growth=True)/g" /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/estimators/run_config.py

cd /opt/tiger/ml-benchmark/tensorflow/tensor2tensor
TF_CONFIG=$(python tf_config.py) t2t-trainer $@ $(python flags.py) --data_dir=$DATA_DIR --output_dir=$ARNOLD_OUTPUT
