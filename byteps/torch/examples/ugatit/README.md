# UGATIT

Our script is based on [Official PyTorch Implementation of UGATIT](https://github.com/znxlwm/UGATIT-pytorch) and you can find the dataset [here](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view).


## Download the dataset

```bash
cd ~/data
gdown 1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF
mkdir selfie2anime && unzip selfie2anime.zip -d selfie2anime
```
The default location of the dataset is ~/data, and the dataset for UGATIT is in ~/data/selfie2anime


## Install dependencies
```bash
sudo apt-get update && sudo apt-get install libgl1 -y
pip3 install opencv-python
```

# How to run
**Note**: Make sure the dataset is in the right location. Set the DMLC_PS_ROOT_URI and ifname in run_espresso.sh and run_baseline.sh.

DMLC_PS_ROOT_URI: the IP address of the root GPU machine

ifname: the network interface card name, e.g., eth0, eth2

## Espresso
Run on each machine
```bash
bash run_espresso.sh WORKERS ID
```
WORKERS: the number of GPU machines in the training

ID: the id of a machine. machines have distinct IDs that start from 0

### An example
Suppose there are four GPU machines, then WORKERS=4 and ID is from 0-3. 
The ID of the root GPU machine is 0.
The command on each GPU machine is
```bash
bash run_espresso.sh 4 ID
```

## Baseline
Run on each machine
```bash
bash run_baseline.sh WORKERS ID
```  

### An example
Suppose there are four GPU machines, then WORKERS=4 and ID is from 0-3. 
The ID of the root GPU machine is 0.
The command on each GPU machine is
```bash
bash run_baseline.sh 4 ID
```
