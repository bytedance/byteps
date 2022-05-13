# GPT-2

## Download the dataset
```bash
mkdir ~/data
cd ~/data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
```
**Note**: the default location of the dataset is ~/data, and the dataset for GPT2 is in ~/data/wikitext-2-raw


## Install dependencies
```bash
bash run_prepare.sh
```

## How to run
**Note**: Make sure the dataset is in the right location. Set the DMLC_PS_ROOT_URI and ifname in run_espresso.sh and run_baseline.sh.

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
