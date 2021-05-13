source common_arnold.sh

if [ "$ARNOLD_ROLE" == "scheduler" ]; then
    export DMLC_ROLE=scheduler
    python3 -c 'import byteps.server as bpss;'
else
    # server
    export DMLC_WORKER_ID=$DMLC_WORKER_ID
    export DMLC_ROLE=server
    python3 -c 'import byteps.server as bpss;' &
    sleep 3

    # worker
    export DMLC_WORKER_ID=$DMLC_WORKER_ID
    export NVIDIA_VISIBLE_DEVICES=0
    export BYTEPS_LOCAL_RANK=$NVIDIA_VISIBLE_DEVICES
    export DMLC_ROLE=worker
    python3 send_recv.py --rank $DMLC_WORKER_ID
fi
