python3 simulator_espresso.py --model vgg16 --node 4 --compressor randomk --cpu --two-level --profile
python3 simulator_espresso.py --model resnet101 --node 4 --compressor dgc --cpu --two-level --profile
python3 simulator_espresso.py --model ugatit --node 4 --compressor dgc --cpu --profile
python3 simulator_espresso.py --model bert --node 4 --compressor randomk --cpu --profile
python3 simulator_espresso.py --model gpt2 --node 4 --compressor efsignsgd --cpu --profile
python3 simulator_espresso.py --model lstm --node 4 --compressor efsignsgd --cpu --two-level --profile
