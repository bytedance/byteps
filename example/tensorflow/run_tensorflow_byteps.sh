#!/bin/bash

if [ "$EVAL_TYPE" == "benchmark" ]; then
  echo "Run synthetic benchmark..."
  python /opt/tiger/byteps/example/tensorflow/synthetic_benchmark.py $@
elif [ "$EVAL_TYPE" == "mnist" ]; then
  echo "Run MNIST ..."
  python /opt/tiger/byteps/example/tensorflow/tensorflow_mnist.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
