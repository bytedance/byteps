#!/bin/bash
export NVIDIA_VISIBLE_DEVICES=0,1
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=1
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000

# set the enriroment variable to diable running performance tests to find the best convolution alglrithm
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

PY_VERSION="3"
path="`dirname $0`"


if [ "$PY_VERSION" = "3" ]; then
	PYTHON="python3"
elif [ "$PY_VERSION" = "2" ]; then	
	PYTHON="python"
else
	echo "Python version error"
fi

BYTEPS_PATH=`${PYTHON} -c "import byteps as bps; path=str(bps.__path__); print(path.split(\"'\")[1])"`
echo "BYTEPS_PATH:${BYTEPS_PATH}" 
# BYTEPS_PATH: /usr/local/lib/python3.6/site-packages/byteps-0.1.0-py3.6-linux-x86_64.egg/byteps/torch

##----------------------------------- 		Modify MXNet 	  ----------------------------------- 
# \TODO huhanpeng: direct get the gradient names in bytePS without modifying MXNet python part
if [ $DMLC_ROLE = "worker" ]; then
	echo "Modify MXNet for workers"
	MX_PATH=`${PYTHON} -c "import mxnet; path=str(mxnet.__path__); print(path.split(\"'\")[1])"`
	echo "MX_PATH: $MX_PATH"
	${PYTHON} $path/../../launcher/insert_code.py \
			--target_file="$MX_PATH/module/executor_group.py" \
			--start="        self.arg_names = symbol.list_arguments()" \
			--end="        self.aux_names = symbol.list_auxiliary_states()" \
			--indent_level=2 \
			--content_str="import os
_param_names = [name for i, name in enumerate(self.arg_names) if name in self.param_names]
path = os.environ.get('TRACE_DIR', '.') + '/' + os.environ.get('BYTEPS_RANK') + '_' + os.environ.get('BYTEPS_LOCAL_RANK') + '/'
if path:
	if not os.path.exists(path):
		os.makedirs(path)
	with open(os.path.join(path, 'arg_namesINpara_names.txt'), 'w') as f:
		for name in _param_names:
			f.write('%s\n' % name) # output execution graph"
else
	echo "No need to modify mxnet for server/scheduler."
fi

## To avoid integrating multiple operators into one single events
# \TODO: may influence the performance
export MXNET_EXEC_BULK_EXEC_TRAIN=0

## install networkx
pip3 install networkx

##----------------------------------- Start to run the program ----------------------------------- 
echo 
echo "-------------------- Start to run the program ---------------"
python $path/../../launcher/launch.py ${PYTHON} $path/train_mnist_byteps.py
# --num-iters 1000
