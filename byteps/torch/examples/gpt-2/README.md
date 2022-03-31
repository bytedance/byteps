
Dataset link: 
https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

# Download the dataset
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
```

# install dependencies
```bash
bash run_prepare.sh
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