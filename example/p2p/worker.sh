source common.sh

export DMLC_WORKER_ID=$1
export BYTEPS_LOCAL_RANK=0
export NVIDIA_VISIBLE_DEVICES=0
export DMLC_ROLE=worker

ENABLE_GLOBAL_RANK=1 python3 send_recv_v2.py --rank $1 --test_mode 0 --cpu $2

