#!/bin/bash

if [ "$EVAL_TYPE" == "imagenet" ]; then
  echo "Run Keras ImageNet ..."
  python /opt/tiger/byteps/example/keras/keras_imagenet_resnet50.py $@
elif [ "$EVAL_TYPE" == "mnist" ]; then
  echo "Run Keras MNIST ..."
  python /opt/tiger/byteps/example/keras/keras_mnist.py $@
elif [ "$EVAL_TYPE" == "mnist_advanced" ]; then
  echo "Run Keras MNIST-advanced ..."
  python /opt/tiger/byteps/example/keras/keras_mnist_advanced.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
