## ===== Single machine ======
## bash launch_p2p_test.sh

## ===== Multiple machines ======
## args: (rank, num_machines)
## Rank-0: bash launch_p2p_test.sh 0 2 
## Rank-1: DMLC_PS_ROOT_URI=ip_of_machine0 bash launch_p2p_test.sh 1 2

set -x
path="`dirname $0`"
rm -f /tmp/socket_*
rm -f /dev/shm/BytePS_*

export BYTEPS_SERVER_DIRECT_RESPONSE=${BYTEPS_SERVER_DIRECT_RESPONSE:=0}
export TOTAL_LEN=${TOTAL_LEN:-4096000}
export BYTEPS_PARTITION_BYTES=40960000
export BYTEPS_LOCAL_SIZE=${BYTEPS_LOCAL_SIZE:=8}

if [ $# -eq 0 ]; then 
    export DMLC_NUM_WORKER=$BYTEPS_LOCAL_SIZE
    export OFFSET=0
    export RANK=0
else 
    export RANK=$1
    export NODE=$2
    export DMLC_NUM_WORKER=$((NODE*8))
    export OFFSET=$((RANK*8))
    echo "Rank is $RANK, total number of machines is $NODE"
fi

pushd $path

function getip() { echo $(ip a show $1 | awk '/inet / {print $2}' |  awk -F'[/]' '{print $1}'); }

if [ "$RANK" == "0" ]; then
    DMLC_NODE_HOST=$(getip eth0) bash run_byteccl_test.sh scheduler &
fi

for i in $(seq $BYTEPS_LOCAL_SIZE)
do
    LOCAL_RANK=$(($i-1))
    x=$(($LOCAL_RANK/2))
    export UCX_NET_DEVICES=${UCX_NET_DEVICES:=mlx5_$x:1}
    CUDA_VISIBLE_DEVICES=$i DMLC_NODE_HOST=$(getip eth$x) bash run_byteccl_test.sh joint $(($OFFSET+$LOCAL_RANK)) test_tensorflow_p2p.py &
done

popd 

wait