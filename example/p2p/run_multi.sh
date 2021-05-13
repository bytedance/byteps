#!/bin/bash

export NCCL_DEBUG="VERSION"

export DMLC_NUM_WORKER=$ARNOLD_WORKER_NUM
export DMLC_NUM_SERVER=$ARNOLD_WORKER_NUM
export DMLC_PS_ROOT_URI=$ARNOLD_WORKER_0_HOST
export DMLC_PS_ROOT_PORT=$ARNOLD_WORKER_0_PORT

export DMLC_ENABLE_RDMA=1
export PS_VERBOSE=2
export PS_KEY_LOG=1
export BYTEPS_P2P_ON=1
export BYTEPS_LOG_LEVEL=TRACE

export BYTEPS_LOCAL_SIZE=$ARNOLD_WORKER_GPU
export BYTEPS_FORCE_DISTRIBUTED=1

if [ "$ARNOLD_ID" == "0" ]; then
    echo "Launch scheduler..."
    # scheduler
    export DMLC_ROLE=scheduler
    python3 -c 'import byteps.server as bpss;' &
fi

# server
export BYTEPS_SERVER_DEBUG=1
export DMLC_ROLE=server
sleep 3
python3 -c 'import byteps.server as bpss;' &

# worker
export BYTEPS_LOCAL_RANK={$BYTEPS_LOCAL_RANK:-0}
export NVIDIA_VISIBLE_DEVICES={$NVIDIA_VISIBLE_DEVICES:-0}
export DMLC_WORKER_ID=$ARNOLD_ID
export DMLC_ROLE=worker
python3 send_recv.py --rank $ARNOLD_ID &

wait
