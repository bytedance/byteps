# Performance Analysis of BytePS

You can analyze the fine-grained performance of BytePS with the profiling tool.

## For Communication Operations

### Usage

Use the following environment variables to enable profiling the communication operations:

``` python
"BYTEPS_TRACE_ON" = "1"
"BYTEPS_TRACE_END_STEP" = "20"
"BYTEPS_TRACE_START_STEP"="10"
"BYTEPS_TRACE_DIR"= "./traces"
```
First `BYTEPS_TRACE_ON` should be set to `1` to enable profiling communication traces. `BYTEPS_TRACE_START_STEP` and `BYTEPS_TRACE_END_STEP` decide the step interval we want to profile, traces from step `BYTEPS_TRACE_START_STEP` to step `BYTEPS_TRACE_END_STEP` steps will be automatically collected and the result traces will be output in the chrome trace format. `BYTEPS_TRACE_DIR` denotes the path where you want to store traces.

The result directory is organized as follows.
```
traces/
├── 0
│   └── comm.json
│ 
└── 1
    └── comm.json
```

Here, `traces/` is the trace directory we defined using `BYTEPS_TRACE_DIR`. `traces/` contains several sub-directories, each of which denotes one GPU and is named with the local rank of this GPU, e.g., path `./traces/0/` stores the traces results of the GPU whose local rank is `0`. Each sub-directory contains following directories/files:
* `comm.json`: the final trace file which contains the communication traces of all gradients;

### Trace Format
Let's look deep into the traces.
``` json
{
    "ph": "X",
    "args": {
        "name": "Comm.byteps.gradient_0"
    },
    "pid": "Comm.byteps.gradient_0",
    "name": "Comm.byteps.gradient_0",
    "ts": 1574685989504865,
    "dur": 24026,
    "tid": "total"
},
{
    "ph": "X",
    "args": {
        "name": "Comm.byteps.gradient_0"
    },
    "pid": "Comm.byteps.gradient_0",
    "name": "Comm.byteps.gradient_0.BROADCAST",
    "ts": 1574685984662375,
    "dur": 1074,
    "tid": "26148864"
}
```
Basically, the trace event format is the same as the standard [Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit). Here, `name` is the name of one event, which can be shown on `chrome://tracing/`. Considering BytePS divides each gradinets to multiple partitions if necessary and each partition needs to go through several types of following operations, namely `QueueType`.
```
  "COORDINATE_REDUCE",
  "REDUCE",
  "COPYD2H",
  "PCIE_REDUCE",
  "COORDINATE_PUSH",
  "PUSH",
  "PULL",
  "COPYH2D",
  "COORDINATE_BROADCAST",
  "BROADCAST"
```
So there are two types of events:
1. If `tid` is `total`, the event records the entire interval to synchronize one gradient, including the queue time. In this case, `name` ends with the gradient index.
2. If `tid` is a number, the event records the interval for each `QueueType` of each partition of one gradient. In this case, `name` ends with the gradient index and the corresponding `QueueType`, `tid` denotes the partition id.

Note that for BytePS, for multiple GPUs on one worker, only the root GPU is responsible for synchronizing with servers, and these GPUs located on one worker update parameters through all-reduce. Therefore, you can observe `PUSH` and `PULL` operations only in the traces of the root GPU. By default, the root GPU is one with the largest local rank.

Below shows a visualization example of `comm.json`.
<img src="https://user-images.githubusercontent.com/17765864/69711658-634e3080-113c-11ea-8d70-fb75f89f2791.png" width="1916">

### Overhead
Below shows the latency when running [`bert_12_768_12`](https://github.com/joapolarbear/gluon-nlp/tree/bert-byteprofile/scripts/bert) model with 2 workers, each containing 2 V100 GPUs with 16GB of memory. BytePS Timeline collects traces during step 10 to step 20 and after step 20, it asynchronously outputs the trace results, which may also cause extra overhead. Ignoring the warm up phase (the first 10 steps), the overhead induced by BytePS Timeline is small.
<img src="https://user-images.githubusercontent.com/17765864/69713426-79a9bb80-113f-11ea-9bec-b588cc051fab.png" width="1916">

