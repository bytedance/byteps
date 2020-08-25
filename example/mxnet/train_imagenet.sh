#!/bin/bash
interface=ens3
ip=$(ifconfig $interface | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
port=1234
NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# training hyper-params
algo=$1
shift
lr=$1
shift
model=resnet50_v2
epochs=120
batch_size=64

# finetune params
threadpool_size=16
omp_num_threads=4
partition_bytes=4096000
min_compress_bytes=1024000
server_engine_thread=4
data_threads=2

# path
repo_path=/home/ubuntu/repos/byteps
worker_hosts=hosts
server_hosts=hosts
script_path=$repo_path/example/mxnet/train_gluon_imagenet_byteps_gc.py
data_path=/home/ubuntu/data/ILSVRC2012/
pem_file=$1
shift

log_file="${algo}-lr${lr}"
compression_args=''
if [[ $algo == "baseline" ]]; then
  threadpool_size=0
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
  compression_args='--compressor randomk --k '${k}' --compress-momentum nesterov'
  log_file=$log_file"-k${k}"
elif [[ $algo == "dithering" ]]; then
  k=$1
  shift
  compression_args='--compressor dithering --k '${k}' --normalize l2'
  log_file=$log_file"-k${k}"
else
  echo "unknown compressor. aborted."
  exit
fi
log_file=$log_file".log"


cmd="python $repo_path/launcher/dist_launcher.py -WH $worker_hosts -SH $server_hosts --scheduler-ip $ip --scheduler-port $port --interface $interface -i $pem_file --username ubuntu --env OMP_WAIT_POLICY:PASSIVE --env OMP_NUM_THREADS:$omp_num_threads --env BYTEPS_THREADPOOL_SIZE:$threadpool_size --env BYTEPS_MIN_COMPRESS_BYTES:$min_compress_bytes --env BYTEPS_NUMA_ON:1 --env NVIDIA_VISIBLE_DEVICES:$NVIDIA_VISIBLE_DEVICES --env BYTEPS_SERVER_ENGINE_THREAD:$server_engine_thread --env BYTEPS_PARTITION_BYTES:$partition_bytes --env BYTEPS_LOG_LEVEL:INFO source ~/.profile; bpslaunch python3 $script_path --model $model --mode hybrid --rec-train $data_path"train.rec" --rec-train-idx $data_path"train.idx" --rec-val $data_path"val.rec" --rec-val-idx $data_path"val.idx" --use-rec --batch-size $batch_size --num-gpus 1 --num-epochs $epochs -j $data_threads --warmup-epochs 5 --warmup-lr $lr --lr $lr --lr-mode cosine $compression_args --logging-file $repo_path/example/mxnet/$log_file"

echo $cmd
exec $cmd
