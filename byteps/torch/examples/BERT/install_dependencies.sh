#!/bin/bash

pip3 install tqdm
pip3 install nvidia-pyindex
pip3 install nvidia-dllogger

# install pre-trained model
cd ./dataset/checkpoint
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_base_qa_squad11_amp/versions/19.09.0/zip -O bert_pyt_ckpt_base_qa_squad11_amp_19.09.0.zip
sudo apt install unzip
unzip bert_pyt_ckpt_base_qa_squad11_amp_19.09.0.zip
cd ../../ && mkdir -p results

# install apex
export http_proxy=http://bj-rd-proxy.byted.org:3128 https_proxy=http://bj-rd-proxy.byted.org:3128 no_proxy=code.byted.org
cd ~
git clone https://github.com/NVIDIA/apex
cd apex
git checkout d6b5ae5d04f531ff862f651e67f241fef88fd159
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./