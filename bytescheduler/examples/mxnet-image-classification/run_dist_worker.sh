./dist_worker.sh 2 2  python train_imagenet.py --network vgg --num-layers 16 --benchmark 1 --kv-store dist_sync --batch-size 32 --disp-batches 100 --num-examples 3520 --num-epochs 1 --gpus 0,1,2,3
