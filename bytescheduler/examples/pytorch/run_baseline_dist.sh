export USE_BYTESCHEDULER=0

horovodrun -np 2 -H localhost:1,172.31.95.237:1 -p 2022 python pytorch_horovod_benchmark.py
