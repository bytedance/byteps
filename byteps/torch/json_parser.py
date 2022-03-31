import json
  
# Opening JSON file
f = open('traces/0/tensor_ready.json',)
  
# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list
time_points = {}
for item in data['TensorReadyTraceEvents']:
    name = item["name"]
    ts = item["ts"]
    if name not in time_points:
        time_points[name] = [ts]
    else:
        time_points[name].append(ts)

iters = 0
for key, value in time_points.items():
    if iters == 0:
        iters = len(value)
    value.sort()


timelines = [[] for _ in range(iters)]
time_gaps = [[0] for _ in range(iters)]

for key, value in time_points.items():
    for i, time in enumerate(value):
        timelines[i].append(time)

for i, timeline in enumerate(timelines):
    timeline.sort()
    for j in range(len(timeline) - 1):
        gap = timeline[j+1] - timeline[j]
        time_gaps[i].append(gap)


# Closing file
f.close()
for time_gap in time_gaps:
    print(sum(time_gap), time_gap)