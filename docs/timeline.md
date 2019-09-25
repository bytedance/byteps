# Performance Analysis of BytePS

You can analyze the performance with the timeline profiled by the servers. It shows the processing duration (from the arrival time to the finish time) of each push/pull request from differnet workers. You might be able to find out the straggler who slows down the training. 


## Usage
Use `export BYTEPS_SERVER_ENABLE_PROFILE=1` to enable the profiling (only valid for servers).

Each server will generate a `server_profile.json` file in current directory. You can also specify the file name and location with `export BYTEPS_SERVER_PROFILE_OUTPUT_PATH=/path/to/your/server_profile.json`

By default it profiles requests of all keys (tensors). Therefore, you may find your `server_profile.json` too large and difficult to analyze. Instead, you can select a specific key ID (from the original `server_profile.json`) to profile: `export BYTEPS_SERVER_KEY_TO_PROFILE=KEY_ID`. For example, if you set `BYTEPS_SERVER_KEY_TO_PROFILE` to `27000832` you will get the following results:

```
......
{"name": "push-9", "ph": "B", "pid": 27000832, "tid": 27000832, "ts": 1569331999234742},
{"name": "push-9", "ph": "E", "pid": 27000832, "tid": 27000832, "ts": 1569331999234778},
{"name": "pull-9", "ph": "B", "pid": 27000832, "tid": 27000832, "ts": 1569331999234878},
{"name": "push-11", "ph": "B", "pid": 27000832, "tid": 27000832, "ts": 1569331999234898},
{"name": "push-11", "ph": "E", "pid": 27000832, "tid": 27000832, "ts": 1569331999234931},
......
```


Then analyze the timeline using `chrome://tracing`.
For example, below shows the profile result of a distributed training case (2 workers and 2 servers). In ps-lite, worker ranks are 9, 11, 13, and etc. So `push-9` and `push-11`  mean the push requests from the first worker and second worker, respectively. From this figure, we can observe that the first worker is slower than the second one. Similarly, you can find whether there is a consistent straggler for large scale training.
![profile](https://user-images.githubusercontent.com/13852819/65565724-53bb3b80-df83-11e9-8490-6bb590d6fd18.png)

