# Reproduction Guide

This is a guide to reproduce our results of gradient compression.

## CIFAR100 

- env: 8 x p3.16xlarge ([p3 instances](https://aws.amazon.com/ec2/instance-types/p3/))
- model: resnet18_v2 

Exec Command
```sh
pssh -i -l ubuntu -H hosts './run.sh onebit'
```
If you are not familiar with `pssh`, please refer to [pssh docs](https://linux.die.net/man/1/pssh).

### Scripts

Replace your scheduler's IP with `192.168.80.230`.

scheduler.sh
```sh
#!/bin/bash --login
export DMLC_NUM_WORKER=8
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=16
export DMLC_PS_ROOT_URI=192.168.80.230
export DMLC_PS_ROOT_PORT=1234
export BYTEPS_LOG_LEVEL=WARNING

nohup bpslaunch >>/dev/null 2>&1 &
```

server.sh
```sh
#!/bin/bash --login
export DMLC_NUM_WORKER=8
export DMLC_ROLE=server
export DMLC_NUM_SERVER=16
export DMLC_PS_ROOT_URI=192.168.80.230
export DMLC_PS_ROOT_PORT=1234
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export BYTEPS_LOG_LEVEL=WARNING

nohup bpslaunch >>/dev/null 2>&1 &
```

worker.sh
```sh
#!/bin/bash --login
which python
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=1
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=2
export DMLC_PS_ROOT_URI=192.168.80.230
export DMLC_PS_ROOT_PORT=1234         
export BYTEPS_LOG_LEVEL=WARNING
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=4
export BYTEPS_THREADPOOL_SIZE=16
export BYTEPS_PARTITION_BYTES=4096000
export BYTEPS_MIN_COMPRESS_BYTES=1024000
export BYTEPS_NUMA_ON=1

if [[ $1 == *"baseline"* ]]; then
  export BYTEPS_THREADPOOL_SIZE=0
  cmd="bpslaunch python example/mxnet/gradient_compression/train_cifar100_byteps_gc.py --model resnet18_v2 --mode hybrib --batch-size 32 --num-gpus 1 --num-epochs 20 -j 2 --warmup-epochs 10 --warmup-lr 0.1 --lr 0.1 --logging-file ${1}"
elif [[ $1 == *"onebit"* ]]; then
  cmd="bpslaunch python example/mxnet/gradient_compression/train_cifar100_byteps_gc.py --model resnet18_v2 --mode hybrib --batch-size 32 --num-gpus 1 --num-epochs 200 -j 2 --warmup-epochs 10 --warmup-lr 0.1 --lr 0.1 --logging-file ${1} --compressor onebit --onebit-scaling --ef vanilla --compress-momentum nesterov"
elif [[ $1 == *"topk"* ]]; then
  cmd="bpslaunch python example/mxnet/gradient_compression/train_cifar100_byteps_gc.py --model resnet18_v2 --mode hybrib --batch-size 32 --num-gpus 1 --num-epochs 200 -j 2 --warmup-epochs 10 --warmup-lr 0.1 --lr 0.1 --logging-file ${1} --compressor topk --k 0.001"
elif [[ $1 == *"randomk"* ]]; then
  cmd="bpslaunch python example/mxnet/gradient_compression/train_cifar100_byteps_gc.py --model resnet18_v2 --mode hybrib --batch-size 32 --num-gpus 1 --num-epochs 200 -j 2 --warmup-epochs 10 --warmup-lr 0.1 --lr 0.1 --logging-file ${1} --compressor randomk --k 0.001"
elif [[ $1 == *"dithering"* ]]; then
  cmd="bpslaunch python example/mxnet/gradient_compression/train_cifar100_byteps_gc.py --model resnet18_v2 --mode hybrib --batch-size 32 --num-gpus 1 --num-epochs 200 -j 2 --warmup-epochs 10 --warmup-lr 0.1 --lr 0.1 --logging-file ${1} --compressor dithering --k 2 --normalize l2"
else
  echo "unsupport compressor"
  exit
fi

if [[ $1 == *"fp16"* ]]; then
  cmd="${cmd} --fp16-pushpull"
fi

cmd="${cmd} 2>&1 >>/dev/null"

echo $cmd

exec $cmd
```

run.sh
```sh
#!/bin/bash --login
ROOT_IP=192.168.80.230

for job in $1; do
  pkill bpslaunch
  pkill python
  echo "start " $job
  ip=`ifconfig ens3 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`
  echo $ip
  if [ $ip == $ROOT_IP ]; then
    ./scheduler.sh
  fi
  ./server.sh
  ./server.sh
  ./worker.sh $job &
  wait $(jobs -p)
  echo $job " done"
  sleep 5
done

clear
rm -rf params lr.s
```

### Results

| Compressor | Val Accu | Time(s) |
| --- | --- | --- | 
| baseline | 72.56% | 419.91 |
| dithering(k=2) | 73.43% | 339.02 |
| dist-EF-SGDM | 72.77% | 380.40 | 
| randomk(k=0.001) | 71.90% | 278.49 |
| topk(k=0.001) | 73.48% | 268.21|


## ImageNet

- env: 8 x p3.16xlarge
- model: resnet50_v2 & VGG16 

Exec Command
```sh
pssh -i -l ubuntu -H hosts './run.sh'
```

### Scripts

Replace your scheduler's IP with `192.168.80.230`.
And replace with your scripts path.


worker.sh for ResNet50_v2
```sh
#!/bin/bash --login
which python3
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=8
export DMLC_ROLE=worker 
export DMLC_NUM_SERVER=16
export DMLC_PS_ROOT_URI=192.168.80.230 
export DMLC_PS_ROOT_PORT=1234        
export BYTEPS_LOG_LEVEL=WARNING
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=4
export BYTEPS_THREADPOOL_SIZE=16
export BYTEPS_PARTITION_BYTES=4096000
export BYTEPS_MIN_COMPRESS_BYTES=1024000
export BYTEPS_NUMA_ON=1

if [ $1 == "baseline" ]; then
  export BYTEPS_THREADPOOL_SIZE=0
  nohup bpslaunch python3 example/mxnet/gradient_compression/train_gluon_imagenet_byteps_gc.py --model resnet50_v2 --mode hybrid --rec-train ~/data/ILSVRC2012/train.rec --rec-train-idx ~/data/ILSVRC2012/train.idx --rec-val ~/data/ILSVRC2012/val.rec --rec-val-idx ~/data/ILSVRC2012/val.idx --use-rec --batch-size 64 --num-gpus 1 --num-epochs 120 -j 2 --warmup-epochs 5 --warmup-lr 0.2 --lr 0.2 --lr-mode cosine --logging-file baseline >>/dev/null 2>&1 &
elif [ $1 == "onebit" ]; then
  nohup bpslaunch python3 example/mxnet/gradient_compression/train_gluon_imagenet_byteps_gc.py --model resnet50_v2 --mode hybrid --rec-train ~/data/ILSVRC2012/train.rec --rec-train-idx ~/data/ILSVRC2012/train.idx --rec-val ~/data/ILSVRC2012/val.rec --rec-val-idx ~/data/ILSVRC2012/val.idx --use-rec --batch-size 64 --num-gpus 1 --num-epochs 120 -j 2 --warmup-epochs 5 --warmup-lr 0.1 --lr 0.1 --lr-mode cosine  --logging-file onebit --compressor onebit --onebit-scaling --ef vanilla --compress-momentum nesterov >>/dev/null 2>&1 &
else
  echo "unsupport compressor"
fi
```


We found VGG16 suffers from the generalization gap with 8 nodes even with full-precision SGDM. So we use 4 nodes for VGG16.

worker.sh for VGG16
```sh
#!/bin/bash --login
which python3
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=4
export DMLC_ROLE=worker 
export DMLC_NUM_SERVER=8
export DMLC_PS_ROOT_URI=192.168.80.230 
export DMLC_PS_ROOT_PORT=1234         
export BYTEPS_LOG_LEVEL=WARNING
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=4
export BYTEPS_THREADPOOL_SIZE=16
export BYTEPS_PARTITION_BYTES=4096000
export BYTEPS_MIN_COMPRESS_BYTES=1024000
export BYTEPS_NUMA_ON=1
 
if [ $1 == "baseline" ]; then  
  export BYTEPS_THREADPOOL_SIZE=0
  nohup bpslaunch python3 example/mxnet/gradient_compression/train_gluon_imagenet_byteps_gc.py --model vgg16 --mode hybrid  --rec-train ~/data/ILSVRC2012/train.rec --rec-train-idx ~/data/ILSVRC2012/train.idx --rec-val ~/data/ILSVRC2012/val.rec --rec-val-idx ~/data/ILSVRC2012/val.idx --use-rec --batch-size 32 --num-gpus 1 --num-epochs 100 -j 2 --warmup-epochs 5 --warmup-lr 0.01 --lr 0.01 --lr-decay-epoch 50,80 --logging-file baseline >>/dev/null 2>&1 &
elif [ $1 == "onebit" ]; then
  nohup bpslaunch python3 example/mxnet/gradient_compression/train_gluon_imagenet_byteps_gc.py --model vgg16 --mode hybrid  --rec-train ~/data/ILSVRC2012/train.rec --rec-train-idx ~/data/ILSVRC2012/train.idx --rec-val ~/data/ILSVRC2012/val.rec --rec-val-idx ~/data/ILSVRC2012/val.idx --use-rec --batch-size 32 --num-gpus 1 --num-epochs 100 -j 2 --warmup-epochs 5 --warmup-lr 0.01 --lr 0.01 --lr-decay-epoch 50,80 --logging-file onebit --compressor onebit --onebit-scaling --ef vanilla --compress-momentum nesterov >>/dev/null 2>&1 &
else
  echo "unsupport compressor"
fi
```

The other scripts are the same, which are skipped here.

### Results

ResNet50_v2
| Compressor | Val Accu | Time(min) |
| --- | --- | --- |
| baseline | 76.91%| 159.04|
| dist-EF-SGDM | 76.32%| 146.27|

VGG16
| Compressor | Val Accu | Time(min) |
| --- | --- | --- |
| baseline | 73.04%| 1171.77|
| dist-EF-SGDM | 72.39%| 636.31|