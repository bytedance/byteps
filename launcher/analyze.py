import os 
import json
import argparse
import networkx as nx


TRACE_DIR = "/Users/huhanpeng/prof/20191009_haibinlin_profiling_debug_rst"

parser = argparse.ArgumentParser(description="Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-s", action="store_true", help="sort the output result")
parser.add_argument("--statistic", type=bool, default=False, help="output statistic results")
parser.add_argument("--graph", type=bool, default=False, help="show the dependency graph")
args = parser.parse_args()

if args.statistic:
	""" Read traces """
	# traces_dir = os.environ.get("TRACE_DIR")
	traces_path= TRACE_DIR + "/bytePS_COMM_110.json"
	with open(traces_path, 'r') as fp:
		_traces = json.load(fp)

	if isinstance(_traces, dict):
		traces = _traces.get("traceEvents")
	elif isinstance(_traces, list):
		traces = _traces
	else:
		raise ValueError("The output file not follow the stardard chrome tracing format!: " + traces_path)

	""" Basic Statistic """
	name2sta = {}
	cat2sta = {}
	for event in traces:
		name = event["name"]
		if name in name2sta:
			name2sta[name]["cnt"] += 1
			name2sta[name]["time"] += event["dur"]
			name2sta[name]["min_t"] = min(name2sta[name]["min_t"], event["dur"])
			name2sta[name]["max_t"] = max(name2sta[name]["max_t"], event["dur"])
		else:
			name2sta[name] = {"cnt": 1, "time": event["dur"], 
				"min_t": event["dur"], "max_t": event["dur"],
				# \TODO huhanpeng: add `cat` field for communication traces
				# "cat": event["cat"] 
				"cat": event["name"].split(".")[0]
				}

	"""calculate the avg """
	for name, statistic in name2sta.items():
		statistic["avg"] = statistic["time"] / statistic["cnt"]
		statistic["var"] = 0.0
		cat = statistic["cat"]
		if cat in cat2sta:
			if statistic["avg"] > cat2sta[cat]["max_t"]:
				cat2sta[cat]["max_t"] = statistic["avg"]
				cat2sta[cat]["max_name"] = name
		else:
			cat2sta[cat] = {"max_t": statistic["avg"], "max_name": name}

	"""calculate the variance"""
	for event in traces:
		name = event["name"]
		name2sta[name]["var"] += pow(event["dur"] - name2sta[name]["avg"], 2)

	for name, statistic in name2sta.items():
		statistic["var"] = statistic["var"] / float(statistic["cnt"])



	""" Output the statistic results """
	# \TODO: device id
	def output(_name2sta):
		print("Profile Statistics.")
		print("===================")
		print("%-30s\t Total Count\t Time (ms)\t Min Time (ms)\t Max Time (ms)\t Avg Time (ms)\t Variance (ms^2)" % "Name")
		print("%-30s\t -----------\t ---------\t -------------\t -------------\t -------------\t ---------------" % "----")
		for name, statistic in _name2sta:
			print("%-30s\t %11d\t %9.4f\t %12.4f\t %13.4f\t %13.4f\t %13.4f" % 
					(name,
					statistic["cnt"],
					statistic["time"] / 1000.0,
					statistic["min_t"] / 1000.0,
					statistic["max_t"] / 1000.0,
					statistic["avg"] / 1000.0,
					statistic["var"] / 1000.0 / 1000.0
					))

	# output(sorted(name2sta.items(), lambda x, y: cmp(x[1]["avg"], y[1]["avg"])))
	if args.s:
		sort_sta = sorted(name2sta.items(), key=lambda x: x[1]["avg"], reverse=True)
	else:
		sort_sta = name2sta.items()
	output(sort_sta)

	# Group by category
	print
	print("Group by category")
	print("===================")
	for cat, statistic in cat2sta.items():
		print("Category: %-10s\t The most time-consuming OP: %-30s -> %13.4f (ms)" % (cat, statistic["max_name"], statistic["max_t"] / 1000.0))


if args.graph:
	mygraph = nx.read_gml(TRACE_DIR + "/dag.gml")
	import matplotlib.pyplot as plt
	# import pylab as plt
	# nx.draw(mygraph, with_labels=False, font_weight='bold')
	# pos = nx.circular_layout(mygraph)
	pos = nx.spectral_layout(mygraph, dim=2, scale=0.5)
	nx.draw(mygraph, pos, with_labels=True, font_size=6)
	plt.show()