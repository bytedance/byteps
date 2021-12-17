#!/usr/bin/env python3

'''
Merge timeline files
Usage:
    python3 merge-timeline.py files/to/merge path/to/dump new_pid timestamp_bias

Example:
say you want to merge 2 json timeline files
    python3 merge-timeline.py a.json,b.json ./merged.json 0,1 0,0

Example:
    python3 merge-timeline.py file01,file02,file03 proc_name01,proc_name02,proc_name03

Example:
    python3 merge-timeline.py a.json,b.json ./merged.json 0,1
a.json will show up as rank 0
b.json will show up as rank 1

Example:
    python3 ../merge-timeline.py 0/comm.json,1/comm.json,2/comm.json,3/comm.json,4/comm.json,5/comm.json,6/comm.json,7/comm.json merged.json
'''
import json
import os, sys
ALIGN_TIME = True
KEEP_PID = False

def obtain_first_align_point(files, event_suffix = ".REDUCE"):
    ret = []
    part_id = None
    for idx, file in enumerate(files):
        with open(file, 'r') as fp:
            traces = json.load(fp)
        if "traceEvents" in traces:
            traces = traces["traceEvents"]
        anchor_ts = None
        for trace in traces:
            if anchor_ts is None:
                anchor_ts = trace["ts"]
            if part_id and trace["tid"] != part_id:
                continue
            if trace["name"].endswith(event_suffix):
                part_id = trace["tid"]
                ret.append(trace["ts"] + trace["dur"] - anchor_ts)
                break
    max_ts = max(ret)
    ret = [max_ts - x for x in ret]
    return ret

def merge_files(files, proc_names, bias, output_path):
    final_traces = []
    for idx, file in enumerate(files):
        with open(file, 'r') as fp:
            traces = json.load(fp)
        if "traceEvents" in traces:
            traces = traces["traceEvents"]
        ts = None
        found = False
        for trace in traces:
            if ALIGN_TIME and ts is None:
                ts = trace["ts"]
            if not KEEP_PID:
                trace["pid"] = proc_names[idx]
            else:
                trace["pid"] = proc_names[idx] + "." + trace["pid"]
            if ALIGN_TIME:
                trace["ts"] = trace["ts"] - ts
            trace["ts"] += bias[idx]
            event_suffix = ".REDUCE"
        final_traces += traces

    with open(output_path, 'w') as fp:
        json.dump(final_traces, fp)

files = sys.argv[1]
output_path = sys.argv[2]
if len(sys.argv) >= 5:
    bias = [float(n)*1000 for n in sys.argv[4].split(",")]
else:
    bias = [0 for _ in files]

files = files.split(",")

if len(files) == 1 and os.path.isdir(files[0]):
    names = sorted(os.listdir(files[0]))
    files = [os.path.join(files[0], n) for n in names]
elif len(sys.argv) < 4:
    names = [str(x) for x in range(len(files))]
else:
    names = sys.argv[3].split(",")

bias = obtain_first_align_point(files, ".REDUCE")
merge_files(files, names, bias, output_path)
