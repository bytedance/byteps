export USE_BYTESCHEDULER=1

horovodrun -np 2 -hostfile myhostfile python pytorch_horovod_benchmark.py
