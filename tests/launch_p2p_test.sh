## ===== Single machine ======
## bash launch_p2p_test.sh

## ===== Multiple machines ======
## args: (rank, num_machines)
## Rank-0: bash launch_p2p_test.sh 0 2 
## Rank-1: DMLC_PS_ROOT_URI=ip_of_machine0 bash launch_p2p_test.sh 1 2

path="`dirname $0`"
rm -f /tmp/socket_*
rm -f /dev/shm/BytePS_*

export BYTEPS_SERVER_DIRECT_RESPONSE=0
export TOTAL_LEN=${TOTAL_LEN:-4096000}
export BYTEPS_PARTITION_BYTES=40960000
export BYTEPS_LOCAL_SIZE=8

if [ $# -eq 0 ]; then 
    export DMLC_NUM_WORKER=8
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
    bash run_byteccl_test.sh scheduler &
fi

CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_0:1 DMLC_NODE_HOST=$(getip eth0) bash run_byteccl_test.sh joint $((OFFSET+0)) test_tensorflow_p2p.py &
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_0:1 DMLC_NODE_HOST=$(getip eth0) bash run_byteccl_test.sh joint $((OFFSET+1)) test_tensorflow_p2p.py &
CUDA_VISIBLE_DEVICES=2 UCX_NET_DEVICES=mlx5_1:1 DMLC_NODE_HOST=$(getip eth1) bash run_byteccl_test.sh joint $((OFFSET+2)) test_tensorflow_p2p.py &
CUDA_VISIBLE_DEVICES=3 UCX_NET_DEVICES=mlx5_1:1 DMLC_NODE_HOST=$(getip eth1) bash run_byteccl_test.sh joint $((OFFSET+3)) test_tensorflow_p2p.py &
CUDA_VISIBLE_DEVICES=4 UCX_NET_DEVICES=mlx5_2:1 DMLC_NODE_HOST=$(getip eth2) bash run_byteccl_test.sh joint $((OFFSET+4)) test_tensorflow_p2p.py &
CUDA_VISIBLE_DEVICES=5 UCX_NET_DEVICES=mlx5_2:1 DMLC_NODE_HOST=$(getip eth2) bash run_byteccl_test.sh joint $((OFFSET+5)) test_tensorflow_p2p.py &
CUDA_VISIBLE_DEVICES=6 UCX_NET_DEVICES=mlx5_3:1 DMLC_NODE_HOST=$(getip eth3) bash run_byteccl_test.sh joint $((OFFSET+6)) test_tensorflow_p2p.py &
CUDA_VISIBLE_DEVICES=7 UCX_NET_DEVICES=mlx5_3:1 DMLC_NODE_HOST=$(getip eth3) bash run_byteccl_test.sh joint $((OFFSET+7)) test_tensorflow_p2p.py &

popd 

wait
