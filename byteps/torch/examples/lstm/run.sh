#!/bin/bash

sudo apt install unzip
bash getdata.sh

bash run_baseline.sh ${ARNOLD_WORKER_NUM} ${DMLC_WORKER_ID}
bash run_bytecomp.sh ${ARNOLD_WORKER_NUM} ${DMLC_WORKER_ID}