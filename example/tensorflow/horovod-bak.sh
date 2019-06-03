#!/bin/bash

service ssh restart

PORT=2222
# TF_ENABLE_RDMA=1

if [ "$ARNOLD_ID" = "0" ];  then

  arr=(${NV_GPU//,/ })
  GPU_NUM=${#arr[@]}

  OLD_IFS="$IFS"
  IFS=","
  arr=($ARNOLD_WORKER_HOSTS)
  IFS="$OLD_IFS"

  WORKER_HOSTS=""
  for s in ${arr[@]}
  do
    IP=${s%:*}
    echo $IP
    `nc -z -v -w5 $IP $PORT`
    result=$?

    while [ "$result" != 0 ]
    do
        `nc -z -v -w5 $IP $PORT`
        result=$?
    done

    WORKER_HOSTS="$IP:$GPU_NUM,$WORKER_HOSTS"
  done
  WORKER_HOSTS=${WORKER_HOSTS:0:-1}

  PROCESS_NUM=$[${#arr[@]}*$GPU_NUM]

  # check https://github.com/uber/horovod/blob/master/docs/running.md
  if [ "$TF_ENABLE_RDMA" != "1" ];  then
      echo "use TCP"
      ARGS=" -mca btl_tcp_if_exclude docker0,lo -mca pml ob1 -mca btl ^openib"
  else
      echo "use RDMA"
      ARGS="  -x PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -mca btl_openib_receive_queues P,65536,256,192,128 --mca pml ob1 --mca btl openib,self,vader --mca btl_openib_cpc_include rdmacm --mca btl_openib_rroce_enable 1"
  fi
  echo $ARGS

  echo "mpirun --allow-run-as-root -np $PROCESS_NUM \
      -H $WORKER_HOSTS \
      -bind-to none -map-by slot \
      -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH $ARGS \
      python /opt/tiger/ml-benchmark/tensorflow/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $@ --data_dir=$DATA_DIR"

  mpirun --allow-run-as-root -np $PROCESS_NUM \
      -H $WORKER_HOSTS \
      -bind-to none -map-by slot \
      -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH $ARGS \
      python /opt/tiger/ml-benchmark/tensorflow/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $@ --data_dir=$DATA_DIR

else
  OLD_IFS="$IFS"
  IFS=","
  arr=($ARNOLD_WORKER_HOSTS)
  IFS="$OLD_IFS"
  IP=$arr
  IP=${IP%:*}
  echo $IP
  `nc -z -v -w5 $IP $PORT`
  result=$?

  while [ "$result" != 0 ]
  do
      `nc -z -v -w5 $IP $PORT`
      result=$?
      echo "connecting..."
  done
  while [ "$result" = 0 ]
  do
      `nc -z -v -w5 $IP $PORT`
      result=$?
      sleep 10
      echo "stopping..."
  done
fi
