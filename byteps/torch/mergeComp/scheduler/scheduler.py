import os
import json
import errno

"""
@comm_type: Communication type
0: FP32
1: FP16
2: intra-node FP16 + inter-node compression + gather + broadcast
3: intra-node FP16 + inter-node compression + alltoall + allgather
4: intra-node FP16 + inter-node compression + allgather
5: intra-node FP16 + CPU compression
6: intra-node compression + intra-node alltoall + inter-node compression + gather + broadcast + intra-node allgather
7: intra-node compression + intra-node alltoall + inter-node compression + alltoall + allgather + intra-node allgather
8: intra-node compression + intra-node gather + inter-node compression + allgather + intra-node broadcast
9: intra-node compression + intra-node alltoall + inter-node compression + allgather + intra-node broadcast
10: allgather
11: alltoall
"""

class Scheduler():
    """
    @filename: the file of scheduler information for each tensor. If not set, FP32 is used for all tensors
    """
    def __init__(self, filename=None, scheduler_type=0, threshold=1024*16):
        self._scheduler_type = scheduler_type
        self._threshold = threshold
        self._map = {}
        self._sizes = {}
        if scheduler_type == -1:
            if ".json" not in filename:
                filename += ".json"

            if filename is not None and os.path.exists(filename):
                with open(filename, 'r') as json_file:
                    data = json.load(json_file)

                for name, option in zip(data["tensor names"], data["options"]):
                    self._map[name] = option
                for name, size in zip(data["tensor names"], data["tensor sizes"]):
                    self._sizes[name] = size
            else:
                print('scheduling file {} does not exist! scheduler type is {}'.format(filename, scheduler_type))

    
    def get_comm_type(self, name, size=2**30):
        if self._scheduler_type == -1:
            if name in self._map:
                return self._map[name]
            else:
                print(f"Warning: tensor {name} not in map")
                return 0

        if self._scheduler_type in (13, 14):
            return self._scheduler_type

        if ((name in self._map) and (self._sizes[name] < self._threshold)) or (size < self._threshold):
            if self._scheduler_type == 5:
                return 14
            else:
                return 0
        else:
            return self._scheduler_type
        

    def print(self):
        print(self._scheduler_type, self._map)