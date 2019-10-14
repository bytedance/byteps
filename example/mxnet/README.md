
## How to run bytePS + MXNet with profiling

Auto profiling is enabeled using the following three enrironment variables.

``` python
my_env["TRACE_ON"] = "ON"
my_env["TRACE_END_STEP"] = "110"
my_env["TRACE_DIR"]= "./traces"
```
`TRACE_ON` is set to `ON` to enable profiling, or the worker runs without profiling. `TRACE_END_STEP` denotes the number of steps we want to profile, byteps Profiler will automatically collect traces for the first `TRACE_END_STEP` steps and output traces in chrome trace format. `TRACE_DIR` denotes the path you want to store the output. You can find the trace result in `$TRACE_DIR/bytePS_COMM_$TRACE_END_STEP.json`.

Currently, only the worker with rank `0` is set to auto profile, you can change the setting by modify `launch.py`.

``` python
if os.environ.get("DMLC_WORKER_ID") == "0" and local_rank == 0:
	...
```

Then, you can run the example code by

``` bash
sh start_mxnet_byteps.sh
```