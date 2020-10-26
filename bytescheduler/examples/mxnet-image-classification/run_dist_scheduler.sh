./dist_scheduler.sh 1 1  python train_imagenet.py --network vgg --num-layers 16 --benchmark 1 --kv-store dist_sync --batch-size 32 --disp-batches 50 --num-examples 1632 --num-epochs 1 --gpus 0
