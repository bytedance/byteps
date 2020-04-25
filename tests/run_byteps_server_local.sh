#!/bin/bash
export PATH=~/.local/bin:$PATH
export DMLC_NUM_WORKER=1
export DMLC_ROLE=$1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234

nohup bpslaunch >>/dev/null 2>&1 &