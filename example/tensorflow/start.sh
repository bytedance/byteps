#!/bin/bash

if [ "$ARNOLD_ROLE" == "server" ]; then
  ARNOLD_ROLE="ps"
fi

echo $ARNOLD_ROLE, $ARNOLD_ID, $ARNOLD_SERVER_HOSTS, $ARNOLD_WORKER_HOSTS, $ARNOLD_OUTPUT
sed -i -e "s/per_process_gpu_memory_fraction=gpu_memory_fraction)/allow_growth=True)/g" /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/estimators/run_config.py

if [ "$EVAL_TYPE" == "cnn" ]; then
  python /opt/tiger/ml-benchmark/tensorflow/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $@ \
    --job_name=$ARNOLD_ROLE --ps_hosts=$ARNOLD_SERVER_HOSTS --worker_hosts=$ARNOLD_WORKER_HOSTS --task_index=$ARNOLD_ID --data_dir=$DATA_DIR --train_dir=${ARNOLD_OUTPUT} --eval_dir ${ARNOLD_OUTPUT}
elif [ "$EVAL_TYPE" == "t2t" ]; then
  cd /opt/tiger/ml-benchmark/tensorflow/tensor2tensor
  echo "TF_CONFIG: " && python tf_config.py
  echo "FLAGS: " && python flags.py
  TF_CONFIG=$(python tf_config.py) t2t-trainer $@ $(python flags.py) --data_dir=$DATA_DIR --output_dir=$ARNOLD_OUTPUT
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
