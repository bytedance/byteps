## How to run byteccl tests

```
export DMLC_PS_ROOT_URI=YOUR_IP
export TEST_TYPE=tensorflow
pkill -9 python3
bash run_byteps_test_byteccl.sh scheduler &
bash run_byteps_test_byteccl.sh joint 0 test_script.py &
bash run_byteps_test_byteccl.sh joint 1 test_script.py &
```
## Example: How to run P2P tests
```
export DMLC_NUM_WORKER=2
export DMLC_PS_ROOT_URI=YOUR_IP
export TEST_TYPE=tensorflow
bash run_byteps_test_byteccl.sh scheduler &
bash run_byteps_test_byteccl.sh joint 0 tensorflow/test_tensorflow_p2p.py &
bash run_byteps_test_byteccl.sh joint 1 tensorflow/test_tensorflow_p2p.py &
```

## Example: How to run allreduce tests
```
export DMLC_NUM_WORKER=2
export DMLC_PS_ROOT_URI=YOUR_IP
export TEST_TYPE=tensorflow
bash run_byteps_test_byteccl.sh scheduler &
bash run_byteps_test_byteccl.sh joint 0 tensorflow/test_tensorflow_cpu_allreduce.py &
bash run_byteps_test_byteccl.sh joint 1 tensorflow/test_tensorflow_cpu_allreduce.py &
```
