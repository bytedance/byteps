export USE_HOROVOD=1
HOROVOD_TIMELINE=/home/net/yrchen/bytescheduler/test/trace/benchmark.json DEBUG_BYTESCHEDULER=0  HOROVOD_CYCLE_TIME=5 \
HOROVOD_READY=0 USE_BYTESCHEDULER=0 BYTESCHEDULER_PARTITION_UNIT=8192000 BYTESCHEDULER_PARTITION_TUNING=0 BYTESCHEDULER_CREDIT=6 mpirun -np 2 \
    --oversubscribe \
    -host localhost \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x USE_BYTESCHEDULER -x HOROVOD_TIMELINE -x HOROVOD_READY -x BYTESCHEDULER_PARTITION_UNIT  -x BYTESCHEDULER_CREDIT  -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib  \
    python3 train.py -data data/multi30k.atok.low.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing -batch_size 512 
