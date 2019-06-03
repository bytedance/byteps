#!/bin/bash

if [ "$EVAL_TYPE" == "mnist" ]; then
    echo "training mnist..."
    python /opt/tiger/byteps/example/pytorch/train_mnist_horovod.py $@
elif [ "$EVAL_TYPE" == "imagenet" ]; then
    echo "training imagenet..."
    python /opt/tiger/byteps/example/pytorch/train_imagenet_resnet50_horovod.py --log-dir ${ARNOLD_OUTPUT} $@
elif [ "$EVAL_TYPE" == "benchmark" ]; then
    echo "running benchmark..."
    python /opt/tiger/byteps/example/pytorch/benchmark_horovod.py $@
elif [ "$EVAL_TYPE" == "transformer" ]; then
    echo "training transformer..."
    cd /opt/tiger/byteps/example/pytorch/transformer && MPIRUN python3 train.py $@
elif [ "$EVAL_TYPE" == "bert" ]; then
    echo "training bert..."
    cd /opt/tiger/byteps/example/pytorch/BERT/ && python3 setup.py install
    cd /opt/tiger/byteps/example/pytorch/BERT/examples && MPIRUN python3 run_classifier.py $@
elif [ "$EVAL_TYPE" == "microbenchmark" ]; then
    echo "running microbenchmark"
    python /opt/tiger/byteps/example/pytorch/microbenchmark-horovod.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
