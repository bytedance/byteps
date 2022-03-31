# Download the dataset
We use `wikitext-2` as our training dataset. 
```bash
#Download wikitext-2
sudo apt install unzip
bash getdata.sh 
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