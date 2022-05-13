# BERT
We fine-tune a pretrained BERT model on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset. Our script is based on [this nvidia repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT). 
The dataset is already in the dataset folder. 
The checkpoint needs to be placed in `./dataset/checkpoint`, which can be downloaded from [NGC](https://ngc.nvidia.com/catalog/models/nvidia:bert_pyt_ckpt_large_qa_squad11_amp/files).


## Install Dependencies
```bash
pip3 install nvidia-pyindex
pip3 install nvidia-dllogger 

git clone https://github.com/NVIDIA/apex
cd apex
git checkout d6b5ae5d04f531ff862f651e67f241fef88fd159
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
**Note**: comment out the version check in apex setup.py if the CUDA version and PyTorch is incompatible

## Dowload model checkpoint
```bash
cd ./dataset/checkpoint
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_base_qa_squad11_amp/versions/19.09.0/zip -O bert_pyt_ckpt_base_qa_squad11_amp_19.09.0.zip
sudo apt install unzip
unzip bert_pyt_ckpt_base_qa_squad11_amp_19.09.0.zip
cd ../../ && mkdir -p results
```

## How to run
**Note**: set the DMLC_PS_ROOT_URI and ifname in run_espresso.sh and run_baseline.sh.

DMLC_PS_ROOT_URI: the IP address of the root GPU machine

ifname: the network interface card name, e.g., eth0, eth2

### Espresso
Run on each machine
```bash
bash run_espresso.sh WORKERS ID
```
WORKERS: the number of GPU machines in the training

ID: the id of a machine. machines have distinct IDs that start from 0

**An example**:
Suppose there are four GPU machines, then WORKERS=4 and ID is from 0-3. 
The ID of the root GPU machine is 0.
The command on each GPU machine is
```bash
bash run_espresso.sh 4 ID
```


### Baselines
Run on each machine
```bash
bash run_baseline.sh WORKERS ID
``` 

**An example**:
Suppose there are four GPU machines, then WORKERS=4 and ID is from 0-3. 
The ID of the root GPU machine is 0.
The command on each GPU machine is
```bash
bash run_baseline.sh 4 ID
```