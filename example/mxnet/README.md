# ByteProfile
An auto profiling tool for distributed machine learning, which is integrated in BytePS. It supports MXNet now. 

**TODO**
* ByteProfile for pytorch has been implemented, need to be integrated into this repository.
* ByteProfile for Tensorflow is under development.
* Whether it can adapt to RDMA network need to be tested further.

## How to collect traces using byteprofile

Auto profiling is enabeled using the following three enrironment variables.

``` python
my_env["TRACE_ON"] = "1"
my_env["TRACE_END_STEP"] = "110"
my_env["TRACE_DIR"]= "./traces"
```
`TRACE_ON` is set to `1` to enable profiling, or the worker runs without profiling. `TRACE_END_STEP` denotes the number of steps we want to profile, byteps Profiler will automatically collect traces for the first `TRACE_END_STEP` steps and output traces in chrome trace format. `TRACE_DIR` denotes the path you want to store traces. 

Then, you can run the example code by

``` bash
sh reproduce.sh
```

~~You can find the trace result in `$TRACE_DIR/bytePS_COMM_${TRACE_END_STEP}step.json`.~~


For each GPU on one worker, the output results are located in the directory of `$TRACE_DIR/${rank}_${local_rank}/`, where `local_rank` denotes the local rank of the card. Basically, it contains following files:

* `arg_namesINpara_names.txt`, it stores the names of gradients which need to be synchronized with the parameter server;
* `bps_trace_local_rank0_3step.json`, it stores the final tracing results, you can further visualize it using `chrome://tracing/`;
* `dag.gml`, a DAG, where each node represents the operator. Note, this is a static DAG, nothing to do with the timeline;
* `temp.json`, temporary outputs of the MXNet profiling tool.

## Trace Format
Let's look deep into the traces, each event in the `bps_trace_local_rank0_3step.json` follows the format as below.
```
	{
            "name": "FW.stage1_unit2_bn2",
            "cat": "operator",
            "ph": "X",
            "ts": 1571030470836169,
            "pid": 96,
            "tid": 14545390205527068052,
            "args": {
                "name": "FW.stage1_unit2_bn2",
                "arg0": "FW.stage1_unit2_conv1",
                "arg1": "Comm.stage1_unit2_bn2_gamma",
                "arg2": "Comm.stage1_unit2_bn2_beta",
                "arg3": "Comm.stage1_unit2_bn2_moving_mean",
                "arg4": "Comm.stage1_unit2_bn2_moving_var"
            },
            "dur": 2498
        },
```
Basically, the trace event formate is the same as the standard format of chrome traceing, you can refer to [Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit) to learn more about this. Here, `name` is the name of the event, which can be shown on `chrome://tracing/`. The event name is composed of operation type and operator real name. Here we define three operator types, `FW. (forward), BW. (backward) and Comm. (communication operation)`. 

Note that we store the dependency information in the `args` field. `args/name` is the node name and `args/arg0, args/arg1, args/arg2, ...` denote the input nodes.


## How to analyze trace results

You can run `analyze.py` to analyze the trace results. A path arg `--path` must be given to specify the file you want to analyze. Another arg `--type` decides the type of analysis you want to process. Currently, it only provides following fetures.

* `--type statistic`, show the statistic results, e.g. 
```bash
python analyze.py --path $TRACE_DIR/0_0/bps_trace_local_rank0_3step.json --type statistic
```
* `--type graph`, visualize the dependency dag graph, e.g.
```bash
python analyze.py --path $TRACE_DIR/0_0/bps_trace_local_rank0_3step.json --type graph
```


