## How to run P2P tests
```
export DMLC_PS_ROOT_URI=YOUR_IP
export TEST_TYPE=tensorflow
bash run_byteps_test_p2p.sh scheduler &
bash run_byteps_test_p2p.sh worker 0 &
bash run_byteps_test_p2p.sh worker 1 &
bash run_byteps_test_p2p.sh server 0 &
bash run_byteps_test_p2p.sh server 1 &
```
