export USE_BYTESCHEDULER=0

horovodrun -np 1 -H localhost:1 python pytorch_horovod_benchmark.py
