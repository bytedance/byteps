#!/bin/bash

path="`dirname $0`"

if [ "$EVAL_TYPE" == "imagenet" ]; then
  echo "Run Keras ImageNet ..."
  python $path/keras_imagenet_resnet50.py $@
elif [ "$EVAL_TYPE" == "mnist" ]; then
  echo "Run Keras MNIST ..."
  python $path/keras_mnist.py $@
elif [ "$EVAL_TYPE" == "mnist_advanced" ]; then
  echo "Run Keras MNIST-advanced ..."
  python $path/keras_mnist_advanced.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
