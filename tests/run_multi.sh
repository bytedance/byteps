# set -x
pkill -9 python3
export UCX_HOME=${UCX_HOME:=/opt/tiger/haibin.lin/ps-lite-test-benchmark/ucx_shm_build}
export DMLC_ENABLE_UCX=${DMLC_ENABLE_UCX:=1}
export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-32}
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-10.212.179.130}

export UCX_CMA_BW=32000MBps 
export UCX_KEEPALIVE_INTERVAL=0 
export UCX_MAX_RNDV_RAILS=1

export UCX_USE_MT_MUTEX="y"
export UCX_IB_NUM_PATHS="2"
export UCX_SOCKADDR_CM_ENABLE="y"
export UCX_RNDV_THRESH="8k"

export BYTEPS_P2P_SKIP_H2D=${BYTEPS_P2P_SKIP_H2D:=1}
export BYTEPS_P2P_SKIP_D2H=1
export BYTEPS_ALLTOALL_FAST_LOCAL=1
export BYTEPS_SERVER_READY_TABLE=1
export BYTEPS_SERVER_ENGINE_THREAD=${BYTEPS_SERVER_ENGINE_THREAD:=4}
export HOSTFILE=${HOSTFILE:="./hosts"}

for LOCAL_RANK in 0 1 2 3
do
  export DMLC_NODE_HOST=$(ip address show  | grep -E "global (eth0|enp1s0f0)" | cut -d ' ' -f 6 | cut -d '/' -f 1)
  export HOST_RANK=$(grep -n $DMLC_NODE_HOST $HOSTFILE | cut -d ':' -f 1)
  # echo $DMLC_NODE_HOST, $HOST_RANK
  export HOST_RANK=$(($HOST_RANK-1))
  export RANK=$((4*$HOST_RANK+$LOCAL_RANK))
  if [ $RANK == "0" ]; then
    CMD="taskset -c 0-31 bash run_byteps_test_p2p.sh scheduler"
    echo "$DMLC_NODE_HOST, $HOST_RANK, $RANK, $CMD"
    if [ $1 != "debug" ]; then
      $CMD &
    fi
  fi
  export BYTEPS_SOCKET_PATH="/tmp/socket_${RANK}"
  mkdir -p $BYTEPS_SOCKET_PATH
  export CORE_START=$((32*$LOCAL_RANK))
  export CORE_END=$((32*$LOCAL_RANK+31))
  CMD="taskset -c $CORE_START-$CORE_END bash run_byteps_test_p2p.sh joint $RANK"
  echo "$DMLC_NODE_HOST, $HOST_RANK, $RANK, $CMD"
  if [ $1 != "debug" ]; then
    $CMD &
  fi
done
wait
