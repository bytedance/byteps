#!/bin/bash

path="`dirname $0`"

if [ "$EVAL_TYPE" == "benchmark" ]; then
  echo "Run synthetic benchmark..."
  python3 $path/synthetic_benchmark.py $@
elif [ "$EVAL_TYPE" == "mnist" ]; then
  echo "Run MNIST ..."
  python3 $path/tensorflow_mnist.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
