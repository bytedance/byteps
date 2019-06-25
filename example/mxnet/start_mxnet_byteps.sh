#!/bin/bash

path="`dirname $0`"

python $path/train_imagenet_byteps.py $@
