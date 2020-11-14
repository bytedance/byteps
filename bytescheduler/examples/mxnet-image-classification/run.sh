./local.sh 1 1  python train_imagenet.py --network resnet --num-layers 18 --benchmark 1 --kv-store dist_sync --batch-size 32 --disp-batches 10 --num-examples 1000 --num-epochs 1 
