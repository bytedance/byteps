set -x
pkill -9 python3
export UCX_HOME=${UCX_HOME:-/opt/tiger/haibin.lin/ps-lite-test-benchmark/ucx_install}
export DMLC_ENABLE_UCX=1 
export DMLC_NUM_WORKER=8
export DMLC_PS_ROOT_URI=${DMLC_PS_ROOT_URI:-10.212.179.138}

export UCX_CMA_BW=32000MBps 
export UCX_KEEPALIVE_INTERVAL=0 
export UCX_MAX_RNDV_RAILS=1

export UCX_USE_MT_MUTEX="y"
export UCX_IB_NUM_PATHS="2"
export UCX_SOCKADDR_CM_ENABLE="y"
export UCX_RNDV_THRESH="8k"


if [ $# -eq 0 ]; then
    export DMLC_NODE_HOST=${DMLC_NODE_HOST:-10.212.179.138}
    taskset -c 0-31   bash run_byteps_test_p2p.sh scheduler &
    mkdir -p /tmp/socket_0
    mkdir -p /tmp/socket_1
    mkdir -p /tmp/socket_2
    mkdir -p /tmp/socket_3
    BYTEPS_SOCKET_PATH="/tmp/socket_0" taskset -c 0-31   bash run_byteps_test_p2p.sh joint 0   &
    BYTEPS_SOCKET_PATH="/tmp/socket_1" taskset -c 32-63  bash run_byteps_test_p2p.sh joint 1   &
    BYTEPS_SOCKET_PATH="/tmp/socket_2" taskset -c 64-95  bash run_byteps_test_p2p.sh joint 2   &
    BYTEPS_SOCKET_PATH="/tmp/socket_3" taskset -c 96-127 bash run_byteps_test_p2p.sh joint 3   &
else
    export DMLC_NODE_HOST=${DMLC_NODE_HOST:-10.212.179.130}
    mkdir -p /tmp/socket_4
    mkdir -p /tmp/socket_5
    mkdir -p /tmp/socket_6
    mkdir -p /tmp/socket_7
    BYTEPS_SOCKET_PATH="/tmp/socket_4" taskset -c 0-31   bash run_byteps_test_p2p.sh joint 4   &
    BYTEPS_SOCKET_PATH="/tmp/socket_5" taskset -c 32-63  bash run_byteps_test_p2p.sh joint 5   &
    BYTEPS_SOCKET_PATH="/tmp/socket_6" taskset -c 64-95  bash run_byteps_test_p2p.sh joint 6   &
    BYTEPS_SOCKET_PATH="/tmp/socket_7" taskset -c 96-127 bash run_byteps_test_p2p.sh joint 7   &
fi
wait
# available: 4 nodes (0-3)
# node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
# node 0 size: 515824 MB
# node 0 free: 494320 MB
# node 1 cpus: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
# node 1 size: 516040 MB
# node 1 free: 499590 MB
# node 2 cpus: 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
# node 2 size: 516080 MB
# node 2 free: 506759 MB
# node 3 cpus: 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
