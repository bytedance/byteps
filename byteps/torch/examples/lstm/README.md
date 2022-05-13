# LSTM

## Download the dataset
```bash
bash getdata.sh 
```
The default location of the dataset is ~/data, and the dataset for LSTM is in ~/data/wikitext-2


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

## Baselines
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