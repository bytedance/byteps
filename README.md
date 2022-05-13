# Espresso


Espresso is a genenral framework built upon BytePS to support compression-enabled data-paralle distributed training. 
It is the first work to holistically consider the search space for gradient compression, such as when to compress tensors, what types of compute resources for compression, and how to choose communication primitives after compression.


## Prerequisites

- CUDA == 11.1
- PyTorch == 1.8.0
- NCCL >= 2.8.3

## Installation

```bash
# In case you need to install PyTorch
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
git clone -b Espresso https://github.com/bytedance/byteps.git
cd byteps
git submodule update --init

export BYTEPS_NCCL_LINK=SHARED
python3 setup.py install --user
cd byteps/torch

pip3 install -r requirements.txt
pip3 install nvidia-pyindex
pip3 install nvidia-dllogger
```

## Examples
The DNN models supported by Espresso are in ./byteps/torch/examples

Following the instructions in each example to reproduce the results.
