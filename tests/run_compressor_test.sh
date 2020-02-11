#!/bin/bash 

IMAGE_NAME="bps-mxnet"
URI="DMLC_PS_ROOT_URI=127.0.0.1"
PORT="DMLC_PS_ROOT_PORT=1234"
CMD="python3 /usr/local/byteps/launcher/launch.py"

if [ $# != 2 ]; then
    echo "wrong input"
    echo "Usage: ./run_compressor_test compressor num_workers"
    exit
fi

valid_compressors=("onebit" "multibit" "topk" "randomk")
ok=0
for cpr in ${valid_compressors[@]}; do
    if [ $cpr == $1 ]; then 
        ok=1
        break
    fi 
done

if [ $ok -eq 0 ]; then 
    echo "unknown compressor named $1"
    exit 
fi

function cleanup {
    echo "clean up"
    docker stop scheduler server
}

trap cleanup EXIT 

rm -f test.log
touch test.log 
echo `date` >> test.log

# launch scheduler
eval "docker run --rm --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name scheduler -e DMLC_NUM_WORKER=$2 -e DMLC_ROLE=scheduler -e DMLC_NUM_SERVER=1 -e ${URI} -e ${PORT} ${IMAGE_NAME} ${CMD} >>/dev/null 2>&1 &" 

# launch server
eval "docker run --rm --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name server -e DMLC_NUM_WORKER=$2 -e DMLC_ROLE=server -e DMLC_NUM_SERVER=1 -e ${URI} -e ${PORT} ${IMAGE_NAME} ${CMD} >>/dev/null 2>&1 &"

echo "=====test begin====="

WORKER_CMD="cd /usr/local/byteps/tests ; python3 /usr/local/byteps/launcher/launch.py python3 -m unittest compressor/test_$1.py"
for ((i=0; i<$2; i++)); do 
    echo "nvidia-docker run --rm --net=host --shm-size=32768m --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name worker${i} -e NVIDIA_VISIBLE_DEVICES=${i} -e DMLC_WORKER_ID=${i} -e DMLC_NUM_WORKER=$2 -e DMLC_ROLE=worker -e DMLC_NUM_SERVER=1 -e ${URI} -e ${PORT} -e BYTEPS_FORCE_DISTRIBUTED=1 ${IMAGE_NAME} bash -c '${WORKER_CMD}' >>test.log 2>&1 &"
    eval "nvidia-docker run --rm --net=host --shm-size=32768m --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name worker${i} -e NVIDIA_VISIBLE_DEVICES=${i} -e DMLC_WORKER_ID=${i} -e DMLC_NUM_WORKER=$2 -e DMLC_ROLE=worker -e DMLC_NUM_SERVER=1 -e ${URI} -e ${PORT} -e BYTEPS_FORCE_DISTRIBUTED=1 ${IMAGE_NAME} bash -c '${WORKER_CMD}' >>test.log 2>&1 &"
done

wait

echo "=====test end====="
echo "test.log:"
cat test.log

