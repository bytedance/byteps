./dist_server.sh 2 2  python train_imagenet.py --network resnet --num-layers 50 --benchmark 1 --kv-store dist_sync --batch-size 128 --disp-batches 10 --num-examples 1280 --num-epochs 1 --gpus 0,1,2,3 --role server
