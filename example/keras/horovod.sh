#!/bin/bash

if [ "$EVAL_TYPE" == "imagenet" ]; then
  echo "Run Keras ImageNet ..."
  python /opt/tiger/byteps/example/keras/keras_imagenet_resnet50.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
