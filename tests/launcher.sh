#!/bin/bash

## ===== Single machine ======
## bash launcher.sh test_tensorflow_p2p.py normal

## ===== Multiple machines ======
## args: (num_machines, rank)
## Rank-0: DMLC_PS_ROOT_URI=ip_of_machine0 bash launcher.sh test_tensorflow_p2p.py normal 2 0
## Rank-1: DMLC_PS_ROOT_URI=ip_of_machine0 bash launcher.sh test_tensorflow_p2p.py normal 2 1

set -x

if [ $# -eq 0 ]; then
    echo "check usage: launcher.sh python_file mode node_num node_rank"
    exit 1
fi
ENTRY=$1
shift

case "$1" in
  simulation)  echo "Test Simulation (two nodes on one physical machine).";
        export MODE=simulation
        ;;
  normal)  echo "Test Normal mode.";
        export MODE=normal
        ;;
  *)    echo "bad option $1";
        exit 1
        ;;
esac
shift

function cleanup() {
    echo "kill all testing processes"
    ps -ef | grep -e "$ENTRY" -e "byteps.server" | grep -v grep | awk '{print $2}' | xargs -r kill -9
    sleep 1
}
trap cleanup EXIT 


path="`dirname $0`"
rm -f /tmp/socket_*
rm -f /dev/shm/BytePS_*
export TOTAL_LEN=${TOTAL_LEN:-4096000}
export BYTEPS_PARTITION_BYTES=40960000
export BYTEPS_LOCAL_SIZE=${BYTEPS_LOCAL_SIZE:=8}

if [ "$MODE" = "simulation" ]; then
    export DMLC_NUM_WORKER=8
    export BYTEPS_LOCAL_SIZE=4
    FOR_NUM=8
else 
    FOR_NUM=$BYTEPS_LOCAL_SIZE
fi

if [ $# -eq 0 ]; then 
    export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-$BYTEPS_LOCAL_SIZE}
    export OFFSET=0
    export RANK=0
elif [ $# -eq 2 ]; then 
    export NODE=$1
    export RANK=$2
    export DMLC_NUM_WORKER=$((NODE*8))
    export OFFSET=$((RANK*8))
    echo "Rank is $RANK, total number of machines is $NODE"
else 
    echo "check usage: launcher.sh python_file mode node_num node_rank"
    exit 1
fi

pushd $path

function getip() { echo $(ip a show $1 | awk '/inet / {print $2}' |  awk -F'[/]' '{print $1}'); }

if [ "$RANK" == "0" ]; then
    DMLC_NODE_HOST=$(getip eth0) bash run_byteccl_test.sh scheduler &
fi

for i in $(seq $FOR_NUM)
do
    LOCAL_RANK=$(($i-1))
    x=$(($LOCAL_RANK/2))
    CUDA_VISIBLE_DEVICES=$i DMLC_NODE_HOST=$(getip eth$x) bash run_byteccl_test.sh joint $(($OFFSET+$LOCAL_RANK)) $ENTRY &
done

popd 

wait