export USE_BYTESCHEDULER=0

horovodrun -np 2 -H 172.31.64.70:1,172.31.66.186:1 python pytorch_horovod_benchmark.py
