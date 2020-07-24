#!/bin/bash

tmux clear-history
clear

export PYTHONPATH=$PYTHOHN_PATH:/usr/local/lib/python3.7/dist-packages

#export BYTEPS_WITHOUT_TENSORFLOW=1
#export BYTEPS_WITH_TENSORFLOW=0
export BYTEPS_WITHOUT_MXNET=1
export BYTEPS_WITH_MXNET=0
export BYTEPS_WITHOUT_PYTORCH=1
export BYTEPS_WITH_PYTORCH=0

#pip3 uninstall byteps; python3 setup.py clean; python3 setup.py develop
# yes | pip3 uninstall byteps; python3 setup.py-tensorflow-only clean; python3 setup.py-tensorflow-only develop

yes | pip3 uninstall byteps; python3 setup.py-tensorflow-only clean;
{ stdbuf -oL -eL  python3 setup.py-tensorflow-only develop; } 2>&1 | tee compile.log
