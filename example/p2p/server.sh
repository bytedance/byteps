source common.sh

export DMLC_WORKER_ID=$1
# export BYTEPS_SERVER_DEBUG=1
export DMLC_ROLE=server
python3 -c 'import byteps.server as bpss;'
