./dist_worker.sh 1 1  python train_imagenet.py --network vgg --num-layers 16 --benchmark 1 --kv-store dist_sync --batch-size 10 --disp-batches 50 --num-examples 500 --num-epochs 1 --gpus 0
