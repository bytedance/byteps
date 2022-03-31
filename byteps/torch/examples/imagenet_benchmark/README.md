
Dataset link: 
# a small dataset from ImageNet
```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagewang.tgz
tar xf imagewang.tgz
```

# how to run
check the dataset is linked in the scripts
Set the DMLC_PS_ROOT_URI and gpus in run_baseline.sh and run_baseline.sh.
WORKERS is the number of machines in the training
ID is the id of a machine. machines have distinct IDs

## ByteComp
Run on each machine
```bash
bash run_bytecomp.sh WORKERS ID
```

## Baseline
Run on each machine
```bash
bash run_baseline.sh WORKERS ID
``` 

# Notes
For VGG16 with PCIe, the compression option for 1-bit quantization is 
options_gpu = [0, 4, 0, 6, 0, 6, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0] for PCIe
options_gpu = [0, 1, 0, 6, 0, 6, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0] for PCIe

For VGG16 with PCIe, the compression option for 1-bit quantization is
[0, 4, 0, 9, 0, 9, 0, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 9, 0, 9, 0, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

replacing the first 4 with 5 results in NCCL hang issue

For VGG16 with PCIe, the compression option for sparsification ratio=0.1% is 
[0, 4, 0, 10, 0, 10, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

For VGG16 with PCIe, the compression option for sparsification ratio=1% is 
[0, 4, 0, 9, 0, 9, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

For VGG16 with NVLink, the compression option for sparsification with compression ratio = 5% is 
[0, 4, 0, 12, 0, 12, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]