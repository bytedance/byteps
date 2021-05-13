pkill python3
set -x
# Replace it with the eth2 IP from: `ip address show`
export DMLC_NODE_HOST=10.188.137.137
export DMLC_PS_ROOT_URI=10.188.136.14
# set DMLC_WORKER_ID = [0, 1] for the two nodes
export DMLC_WORKER_ID=0
export DMLC_PS_ROOT_PORT=9998
# turn these off during perf test
export BYTEPS_LOG_LEVEL=TRACE
export PS_VERBOSE=2
export PS_KEY_LOG=1
export BYTEPS_SERVER_DEBUG=1
# keep the remaining ones unchanged
export DMLC_ENABLE_RDMA=1
export NCCL_DEBUG="VERSION"
export DMLC_INTERFACE=eth2
export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=2
export BYTEPS_P2P_ON=1
export BYTEPS_FORCE_DISTRIBUTED=1

# scheduler
if [ "$DMLC_WORKER_ID" == "0" ]; then
    export DMLC_ROLE=scheduler
    python3 -c 'import byteps.server as bpss;' &
fi
# server
export DMLC_ROLE=server
sleep 3; python3 -c 'import byteps.server as bpss;' &
# worker 
export BYTEPS_LOCAL_SIZE=1
export BYTEPS_LOCAL_RANK=0
export NVIDIA_VISIBLE_DEVICES=0
export DMLC_ROLE=worker
python3 benchmark.py --rank $DMLC_WORKER_ID --niter 1 --size 1024000 &
# in case we need to debug the worker
# gdb -ex run --args python3 benchmark.py --rank $DMLC_WORKER_ID --niter 20 --size 1024001
wait
