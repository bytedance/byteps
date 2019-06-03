#!/bin/bash

if [ "$EVAL_TYPE" == "benchmark" ]; then
  echo "Run synthetic benchmark..."
  python /opt/tiger/byteps/example/tensorflow/horovod/synthetic_benchmark.py $@
elif [ "$EVAL_TYPE" == "perseus" ]; then
  echo "Run perseus synthetic benchmark..."
  python /opt/tiger/byteps/example/tensorflow/horovod/synthetic_benchmark_perseus.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
