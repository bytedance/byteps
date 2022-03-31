for model in "vgg16" "resnet101" "bert" "gpt2" "lstm"
do
    for compressor in "randomk" "dgc" "efsignsgd" "onebit"
    do
        for cpu in "" "--cpu"
        do
            python simulator.py --model=${model} --compressor ${compressor} --nvlink ${cpu}
            python simulator.py --model=${model} --compressor ${compressor} ${cpu}
            python simulator.py --model=${model} --compressor ${compressor} --nvlink ${cpu} --two-level
            python simulator.py --model=${model} --compressor ${compressor} ${cpu} --two-level
        done
    done
done
