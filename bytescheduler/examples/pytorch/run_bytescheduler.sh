export USE_BYTESCHEDULER=1

horovodrun -np 1 -H 172.31.64.70:1 python pytorch_horovod_benchmark.py
