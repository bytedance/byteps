#!/bin/bash
interface=eth2
ip=$(ifconfig $interface | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
port=1234
NVIDIA_VISIBLE_DEVICES=0,1

# training hyper-params
algo=$1
shift
lr=$1
shift
epochs=90
batch_size=64

# finetune params
threadpool_size=16
omp_num_threads=4
partition_bytes=4096000
min_compress_bytes=1024000
server_engine_thread=4
data_threads=2

# path
repo_path=$HOME/repos/byteps
worker_hosts=worker-hosts
server_hosts=server-hosts
script_path=$repo_path/example/pytorch/train_imagenet_resnet50_byteps.py
data_path=$HOME/data/ILSVRC2012/
pem_file=$1
shift

log_file="${model}-${algo}-lr${lr}"
compression_args=''
if [[ $algo == "baseline" ]]; then
  threadpool_size=0
  fp16=$1
  shift
  if [[ $fp16 == "1" ]]; then
    echo "fp16"
    compression_args='--fp16-pushpull'
  fi
elif [[ $algo == "onebit" ]]; then
  compression_args='--compressor onebit --onebit-scaling --ef vanilla --compress-momentum nesterov'
elif [[ $algo == "topk" ]]; then
  k=$1
  shift
  compression_args='--compressor topk --k '${k}' --ef vanilla --compress-momentum nesterov'
  log_file=$log_file"-k${k}"
elif [[ $algo == "randomk" ]]; then
  k=$1
  shift
  compression_args='--compressor randomk --k '${k}' --ef sparse --compress-momentum nesterov'
  log_file=$log_file"-k${k}"
elif [[ $algo == "dithering" ]]; then
  k=$1
  shift
  compression_args='--compressor dithering --k '${k}' --partition natural --normalize max'
  log_file=$log_file"-k${k}"
else
  echo "unknown compressor. aborted."
  exit
fi
log_file=$log_file"$1.log"
shift


data_config="--train-dir $data_path/train --val-dir $data_path/val"


cmd="python $repo_path/launcher/dist_launcher.py -WH $worker_hosts -SH $server_hosts --scheduler-ip $ip --scheduler-port $port --interface $interface -i $pem_file --username yuchen --env OMP_WAIT_POLICY:PASSIVE --env OMP_NUM_THREADS:$omp_num_threads --env BYTEPS_THREADPOOL_SIZE:$threadpool_size --env BYTEPS_MIN_COMPRESS_BYTES:$min_compress_bytes --env BYTEPS_NUMA_ON:1 --env NVIDIA_VISIBLE_DEVICES:$NVIDIA_VISIBLE_DEVICES --env BYTEPS_SERVER_ENGINE_THREAD:$server_engine_thread --env BYTEPS_PARTITION_BYTES:$partition_bytes --env BYTEPS_LOG_LEVEL:INFO  source ~/.zshrc; bpslaunch python3 $script_path  $data_config --batch-size $batch_size  --epochs $epochs --warmup-epochs 5  --base-lr $lr $compression_args"

echo $cmd
exec $cmd
