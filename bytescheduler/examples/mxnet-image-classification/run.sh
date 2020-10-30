 ./local.sh 1 1  python train_imagenet.py --network vgg --num-layers 16 --benchmark 1 --kv-store dist_sync --batch-size 32 --disp-batches 10 --num-examples 1920 --num-epochs 1 --gpus 0
