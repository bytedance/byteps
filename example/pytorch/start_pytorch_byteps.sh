#!/bin/bash

if [ "$EVAL_TYPE" == "mnist" ]; then
    echo "training mnist..."
    python /opt/tiger/byteps/example/pytorch/train_mnist_byteps.py $@
elif [ "$EVAL_TYPE" == "imagenet" ]; then
    echo "training imagenet..."
    python /opt/tiger/byteps/example/pytorch/train_imagenet_resnet50_byteps.py --log-dir $@
elif [ "$EVAL_TYPE" == "benchmark" ]; then
    echo "running benchmark..."
    python /opt/tiger/byteps/example/pytorch/benchmark_byteps.py $@
elif [ "$EVAL_TYPE" == "microbenchmark" ]; then
    echo "running microbenchmark"
    python /opt/tiger/byteps/example/pytorch/microbenchmark-byteps.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
