./dist_server.sh 1 1  python train_imagenet.py --network resnet --num-layers 50 --benchmark 1 --kv-store dist_sync --batch-size 32 --disp-batches 500 --num-examples 16320 --num-epochs 1 --gpus 0
