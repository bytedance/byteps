#!/bin/bash

# before running this, should make sure the data has been downloaded from:
# https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

export DATA_DIR=${DATA_DIR:-~/data}
export TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/wikitext-2-raw/wiki.train.raw}
export TEST_FILE=${TRAIN_FILE:-$DATA_DIR/wikitext-2-raw/wiki.test.raw}
export DISTRIBUTED_FRAMEWORK=${DISTRIBUTED_FRAMEWORK:-byteps}

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ~ 
cd $THIS_DIR/gpt-2
pip3 install .
pip3 install -r ./examples/requirements.txt

# export NCCL_P2P_DISABLE=1

# for the first run, you can do a dryrun to
# prepare/download necessary configurations and models
# in order to avoid multi-write conflicts 
cd $THIS_DIR/gpt-2/examples
python3 -m torch.distributed.launch --nproc_per_node 8 run_lm_finetuning.py \
    --output_dir=output  \
    --model_type=gpt2   \
    --model_name_or_path=gpt2 \
    --do_train   \
    --save_steps 1000000     \
    --overwrite_output_dir    \
    --num_train_epochs 1     \
    --per_gpu_eval_batch_size 4     \
    --train_data_file=$TRAIN_FILE