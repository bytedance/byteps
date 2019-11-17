#!/bin/bash

path="`dirname $0`"

python $path/../../launcher/launch.py ${PYTHON} $path/train_imagenet_byteps.py --benchmark 1 --batch-size=32 