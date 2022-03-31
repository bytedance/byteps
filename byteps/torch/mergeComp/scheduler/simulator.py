import copy
import pickle
import json
import sys, os
import errno
from argparse import ArgumentParser
from model_tensor import *

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
"""

"""
A simulator to run the scheduling algorithm with GPU compression only for a given hardware setup and a model.
The return value is the compression decision for each tensor.
@comm_type: Communication type
0: FP32
1: FP16
2: intra-node FP16 + compression
3: two-level compression
4: intra-node FP16 + CPU compression
"""
class Simulator():
    # TODO: ready_time as the input or grad_gap_time as the input?
    """
    @tensor_sizes: the number of elements in each tensor
    @ready_time_gap: the gap between the ready time of gradient tensors during BP. The zero point the ready time of the first tensor. The time unit is us.
    @comm_overhead: the communication overhead given a tensor and its compression option
    @comp_overhead: the compression overhead given a tensor and its compression option
    @comp_ratio: the compression ratio
    @comp_type: the category of the compression algorithm. False indicates quantization and True indicates sparsification
    @options_num: the number of options
    """
    def __init__(self, tensor_sizes, ready_time_gap, comm_overheads, comp_overheads, embedding_tensors, args, options_num=4):
        self._tensors_num = len(tensor_sizes)
        self.set_tensor_sizes(tensor_sizes)

        assert(self._tensors_num == len(ready_time_gap))
        # make sure that the first gap is 0
        assert(ready_time_gap[0] == 0)
        self._ready_time_gap = ready_time_gap
        self.set_ready_time()

        self._comm_overheads = comm_overheads
        self._comp_overheads = comp_overheads
        self._embedding_tensors = embedding_tensors
        self._model_name = args.model
        self._compressor = args.compressor
        self._comp_type = True if args.compressor in ["randomk", "dgc"] else False
        self._options_num = options_num
        self._gpus_per_node = args.gpus_per_node
        self._nodes = args.nodes
        self._comp_ratio = 1 / 32 if not self._comp_type else args.comp_ratio * 2
        self._cpu = args.cpu
        self._two_level = args.two_level
        self._pcie = not args.nvlink

        self.set_thresholds()

        self._options = [0] * self._tensors_num
        self._flags = [False] * self._tensors_num
        
        # the starting tensor index of the last communication group for different compression options
        self._last_comm_groups = [0] * self._options_num
        
    
    def set_tensor_sizes(self, tensor_sizes):
        self._tensor_sizes = []
        self._tensor_names = []
        for tensor in tensor_sizes:
            name, size = tensor
            self._tensor_names.append(name)
            self._tensor_sizes.append(size)
        print(self._tensor_sizes)


    def get_stat(self):
        large_tensor_thres = 2 ** 19
        large_tensors_num = 0
        for size in self._tensor_sizes:
            if size > large_tensor_thres:
                large_tensors_num += 1

        print("\tmodel: {}\n\tmodel size: {}\n\ttensor number: {}\n\tlarge tensors:{}".format(
            self._model_name,
            sum(self._tensor_sizes),
            len(self._tensor_sizes),
            large_tensors_num
        ))

    def set_ready_time(self):
        self._ready_time = []
        ready_time = 0
        for t in self._ready_time_gap:
            ready_time += t
            self._ready_time.append(ready_time)
        self._base_overhead = self._ready_time[-1]


    def set_thresholds(self):
        self._thresholds = [0] * self._options_num
        
        # dummy threshold for FP32
        self._thresholds[0] = 0
        # TODO: dynamically set the thresholds. 
        # these thresholds are critical to determine the optimal performance of compression
        # FP16 if tensor size > self._thresholds[1], note the message size is tensor size * 4
        self._thresholds[1] = 2**17
        if self._pcie:
            self._thresholds[2] = 2**20
            self._thresholds[3] = 2**23
        else:
            # for BERT, self._thresholds[2] = 2**19
            self._thresholds[2] = 2**20
            self._thresholds[3] = 2**22


    def set_final_comm(self, options=None):
        if options is None:
            options = copy.deepcopy(self._options)
        # TODO: how to set the threshold
        threshold = 2**24
        for i in range(len(options)):
            # one-level compression 
            if options[i] == 2:
                if self._comp_type == True:
                    # sparsification
                    options[i] = 4
                else:
                    # quantization
                    if self._tensor_sizes[i] < threshold: #or self._nodes <= 3:
                        options[i] = 4
                    else:
                        options[i] = 2
            elif options[i] == 3:
                # two-level compression
                if self._comp_type and self._comp_ratio < 0.005:
                    options[i] = 10
                elif self._comp_type:
                    options[i] = 9
                else:
                    options[i] = 6
            elif options[i] == 4: 
                # CPU compression
                options[i] = 5
        return options


    def get_comm_overhead(self, tensor_size, option=0):
        # TODO: output the communication overhead based on the input tensor size
        # alpha is the latency, beta is the communication time of one unit (1KB)
        # alpha, beta1, beta2 = 150, 0.02, 0.8
        alpha, beta1, beta2 = 150, 0.02, 0.8
        if self._pcie:
            beta1 = 0.4
        message_size = tensor_size * 4
        k = self._gpus_per_node
        N = self._nodes
        if option == 0:
            # FP32
            intra_comm_units = 2 * (k-1)/k * message_size // 1000
            inter_comm_units = 2 * (N-1)/N * message_size // 1000
            return 2*alpha + beta1 * intra_comm_units + beta2 * inter_comm_units
        elif option == 1:
            # FP16
            message_size = message_size // 2
            intra_comm_units = 2 * (k-1)/k * message_size // 1000
            inter_comm_units = 2 * (N-1)/N * message_size // 1000
            return 2*alpha + beta1 * intra_comm_units + beta2 * inter_comm_units
        elif option == 2:
            # Compress
            intra_comm_units = 2 * (k-1)/k * message_size / 2 // 1000
            inter_comm_units = 2 * (N-1)/N * message_size * self._comp_ratio // 1000
            return 2*alpha + beta1 * intra_comm_units + beta2 * inter_comm_units
        elif option == 3:
            # two-level compression
            # intra-node alltoall + inter-node alltoall + inter-node allgather + intra-node allgather
            intra_comm_units = 2 * (k-1)/k * message_size * self._comp_ratio // 1000
            inter_comm_units = 2 * (N-1)/N * message_size * self._comp_ratio // 1000
            return 4*alpha + beta1 * intra_comm_units + beta2 * inter_comm_units
        return 0

    
    def get_compression_overhead(self, tensor_size, option=0):
        # 8MB
        base_size = 2**20
        # after intra-node communication (reduce-scatter), the tensor size becomes 1/8 of the original tensor size
        tensor_size /= self._gpus_per_node
        base_overhead = self._comp_overheads[0][option]
       
        if tensor_size <= base_size:
            return base_overhead
        else:
            return base_overhead * tensor_size/base_size


    def get_encoding_overhead(self, tensor_size, option=0):
        return self.get_compression_overhead(tensor_size, option) - self.get_decoding_overhead(tensor_size, option)

    
    def get_decoding_overhead(self, tensor_size, option=0):
        # 8MB
        base_size = 2**20
        # base_overhead = 0
        # after intra-node communication (reduce-scatter), the tensor size becomes 1/8 of the original tensor size
        tensor_size /= self._gpus_per_node
        base_overhead = self._comp_overheads[1][option]
       
        if tensor_size <= base_size:
            return base_overhead #* self._nodes
        else:
            return base_overhead * tensor_size/base_size #* self._nodes


    def get_cpu_compression_overhead(self, tensor_size):
        # set 40 for BERT and 8 for VGG16
        # bw = 8 if self._comp_type else 6
        bw = self._comp_overheads[2][0]
        message_size = tensor_size * 4
        return message_size * 8 / (bw*1000)


    def get_fp32_comm_finish_time(self):
        # the first element is a dummy one
        comm_finish_time = [0] * (self._tensors_num + 1)

        for i in range(self._tensors_num):
            start_time = comm_finish_time[i]
            if self._ready_time[i] >= start_time:
                # there is a communication gap between tensor i and tensor i-1
                start_time = self._ready_time[i]
                self._last_comm_groups[0] = i
            comm_finish_time[i+1] = start_time + self.get_comm_overhead(self._tensor_sizes[i])

        for i in range(self._last_comm_groups[0]):
            self._flags[i] = True

        print("fp32 last comm group", self._last_comm_groups[0])

    # reset the starting point for the timeline
    # @start_pos: calibrate the ready time from the start_pos_{th} tensor
    def calibrate_ready_time(self, ready_time, latency, start_pos=0):
        for i in range(start_pos, len(ready_time)):
            ready_time[i] += latency


    # If the tensor size is larger than a threshold, set the option
    def set_tensors_option(self, tensor_sizes, options, option):
        for i in range(len(tensor_sizes)):
            for j in range(option+1):
                if tensor_sizes[i] > self._thresholds[j]:
                    options[i] = j


    def set_tensor_option(self, tensor_size):
        if tensor_size > self._thresholds[3] and self._two_level:
            return 3
        elif tensor_size > self._thresholds[2]:
            return 2
        elif tensor_size > self._thresholds[1]:
            return 1
        else:
            return 0


    def get_comm_finish_time(self, option):
        last_comm_group = self._last_comm_groups[option-1]

        tensors_num = self._tensors_num - last_comm_group 
        tensor_sizes = self._tensor_sizes[last_comm_group:]
        options = self._options[last_comm_group:]
        self.set_tensors_option(tensor_sizes, options, option)
        ready_time = self._ready_time[last_comm_group:]
        self.calibrate_ready_time(ready_time, -ready_time[0])
        self._options[last_comm_group:] = options

        # the first element is a dummy one
        comm_finish_time = [0] * (tensors_num + 1)
        
        for i in range(tensors_num):
            # if we need to compress the tensor, delay the training process
            encoding_overhead = self.get_encoding_overhead(tensor_sizes[i], options[i])
            self.calibrate_ready_time(ready_time, encoding_overhead, i)

            start_time = comm_finish_time[i]
            if ready_time[i] >= start_time:
                # there is a communication gap between tensor i and tensor i-1
                start_time = ready_time[i]
                self._last_comm_groups[option] = i + last_comm_group

            comm_finish_time[i+1] = start_time + self.get_comm_overhead(tensor_sizes[i], options[i])

        print("last comm group for option {} is {}".format(option, self._last_comm_groups[option]))


    def calculate_total_overhead_cpu(self, tensor_sizes, ready_time, options):
        """
        CPU overhead occurs after intra-node communication.
        Intra-node communication -> Worker CPU encoding -> PUSH -> Server CPU decoding and encoding 
        -> PULL -> Worker CPU decoding -> Intra-node communication
        """
        assert(len(tensor_sizes) == len(options))
        num = len(tensor_sizes)
        comm_finish_time = [0] * (num + 1)
        cpu_finish_time = 0
        ready_time_overhead = copy.deepcopy(ready_time)
        for i in range(num):
            if options[i] < 4:
                encoding_overhead = self.get_encoding_overhead(tensor_sizes[i], options[i])
                self.calibrate_ready_time(ready_time_overhead, encoding_overhead, i)
                start_time = max(comm_finish_time[i], ready_time_overhead[i])
                # TODO: if we need to consider the compression overhead as the part of the communication time
                compression_overhead = 0
                if options[i] >= 2:
                    compression_overhead = self.get_compression_overhead(tensor_sizes[i], options[i])
                comm_finish_time[i+1] = start_time + compression_overhead + self.get_comm_overhead(tensor_sizes[i], options[i])
            elif options[i] == 4:
                # in the pipeline, the bottleneck of CPU compression is the encoding and decoding.
                cpu_start_time = max(comm_finish_time[i], ready_time_overhead[i], cpu_finish_time)
                cpu_finish_time = cpu_start_time + self.get_cpu_compression_overhead(tensor_sizes[i]) + self.get_comm_overhead(tensor_sizes[i], 2)
                comm_finish_time[i+1] = comm_finish_time[i]

        #print(options, comm_finish_time[-1], cpu_finish_time)
        return max(comm_finish_time[-1], cpu_finish_time)

    
    def get_total_overhead(self, options=None):
        if options is None:
            options = self._options
        return self.calculate_total_overhead_cpu(self._tensor_sizes, self._ready_time, options)


    def get_base_overhead(self):
        return self.get_total_overhead()


    def search_options(self, option):
        last_comm_group = self._last_comm_groups[option-1]

        tensors_num = self._tensors_num - last_comm_group 
        tensor_sizes = self._tensor_sizes[last_comm_group:]
        if len(tensor_sizes) == 0:
            return self._options

        options = self._options[last_comm_group:]
        ready_time = self._ready_time[last_comm_group:]
        best_overhead = self.calculate_total_overhead_cpu(tensor_sizes, copy.deepcopy(ready_time), options)

        # prioritize the tensors in the embedding layers
        for idx in self._embedding_tensors:
            tmp = options[idx]
            option = self.set_tensor_option(tensor_sizes[idx])
            if option > 1:
                options[idx] = option
            else:
                continue

            overhead = self.calculate_total_overhead_cpu(tensor_sizes, copy.deepcopy(ready_time), options)
            if overhead < best_overhead:
                best_overhead = overhead
            else:
                options[idx] = tmp


        for idx in range(tensors_num):
            tmp = options[idx]
            option = self.set_tensor_option(tensor_sizes[idx])
            if option > 1:
                options[idx] = option
            else:
                continue

            overhead = self.calculate_total_overhead_cpu(tensor_sizes, copy.deepcopy(ready_time), options)
            if overhead < best_overhead:
                best_overhead = overhead
            else:
                options[idx] = tmp

            if tensor_sizes[idx] < 2 ** 24 and self._cpu:
                options[idx] = 4
                overhead = self.calculate_total_overhead_cpu(tensor_sizes, copy.deepcopy(ready_time), options)
                if overhead < best_overhead:
                    best_overhead = overhead
                else:
                    options[idx] = 2

        self._options[last_comm_group:] = options


    def set_model_options(self):
        self.get_fp32_comm_finish_time()
        print("FP32 overhead:", self.get_total_overhead())
        self.get_comm_finish_time(option=1)
        print("FP16 overhead:", self.get_total_overhead())
        self.search_options(option=2)
        print("compression overhead:", self.get_total_overhead())
        return self._options

    """
    The format of the saved file with pickle is: 
    tensor names \n
    tensor options 
    """
    def save_options(self, filename, options=None):
        if options is None:
            options = self._options

        with open(filename, 'wb') as f:
            pickle.dump(self._tensor_names, f)
            pickle.dump(options, f)
            pickle.dump(self._tensor_sizes, f)


    def load_options(self, filename):
        with open(filename, 'rb') as f:
            tensor_names = pickle.load(f)
            options = pickle.load(f)
            tensor_sizes = pickle.load(f)
        return tensor_names, options, tensor_sizes


    def update_options(self, filename, options=None):
        if options is None:
            return 
        with open(filename, 'rb') as f:
            tensor_names = pickle.load(f)
            pickle.load(f)
            tensor_sizes = pickle.load(f)

        with open(filename, 'wb') as f:
            pickle.dump(tensor_names, f)
            pickle.dump(options, f)
            pickle.dump(tensor_sizes, f)

    """
    dump and load the compression information with json
    """
    def json_dump_options(self, filename, options=None):
        if options is None:
            options = self._options

        data = {}
        data["model"] = self._model_name
        data["compressor"] = self._compressor
        data["nodes"] = self._nodes
        data["gpus_per_node"] = self._gpus_per_node
        data["comp_ratio"] = self._comp_ratio
        data["hierarchical comm"] = self._two_level
        data["CPU compression"] = self._cpu
        data["PCIe"] = self._pcie
        data["tensor names"] = self._tensor_names
        data["tensor sizes"] = self._tensor_sizes
        data["options"] = options

        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=2)


    def json_load_options(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data


    def json_update_options(self, filename, options=None):
        if options is None:
            return 

        if not os.path.exists(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            data["options"] = options
            json.dump(data, json_file, indent=2)
        
        return data


def parse_args():
    parser = ArgumentParser(description="ByteComp Simulator")
    parser.add_argument('--nvlink', action='store_true', default=False,
                        help='PCIe for intra-node communication. False indicates NVLink for intra-node communication')
    parser.add_argument('--two-level', action='store_true', default=False,
                        help='use two-level compression')
    parser.add_argument('--model', type=str, default="vgg16",
                        help='model for simulation')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='CPU is used for compression')
    parser.add_argument('--compressor', type=str, default="randomk",
                        help='True for sparsification and false for quantization')
    parser.add_argument('--comp-ratio', type=float, default=0.01,
                        help='the compression ratio')
    parser.add_argument('--bandwidth', type=int, default=25,
                        help='network bandwidth for inter-node communication')
    parser.add_argument('--nodes', type=int, default=2,
                        help='the number of nodes')
    parser.add_argument('--gpus-per-node', type=int, default=8,
                        help='the number of GPUs on each node')
    parser.add_argument('--read-json', type=str, default="",
                        help='read the scheduling info from file')
    return parser.parse_args()



def init_simulator(args):
    model_name = args.model
    compressor = args.compressor
    comp_overhead = [[0, 0, 0, 0], [0, 0, 0, 0]]
    if compressor == "randomk":
        comp_overhead = random_comp_overhead
    elif compressor == "dgc":
        comp_overhead = dgc_comp_overhead
    elif compressor == "efsignsgd":
        comp_overhead = efsignsgd_comp_overhead
    elif compressor == "onebit":
        comp_overhead = onebit_comp_overhead
    else:
        print("The compressor info of {} is not profiled yet".format(model_name))
        sys.exit()

    if model_name == "vgg16":
        return Simulator(vgg16_tensor_sizes, vgg16_time_gap, None, comp_overhead, [], args)
    elif model_name == "resnet101":
        return Simulator(resnet101_tensor_sizes, resnet101_time_gap, None, comp_overhead, [], args)
    elif model_name == "gpt2":
        return Simulator(gpt2_tensor_sizes, gpt2_time_gap, None, comp_overhead, gpt2_embedding_tensors, args)
    elif model_name == "bert":
        return Simulator(bert_tensor_sizes, bert_time_gap, None, comp_overhead, bert_embedding_tensors, args)
    elif model_name == "lstm":
        return Simulator(LSTM_tensor_sizes, LSTM_time_gap, None, comp_overhead, [], args)
    else:
        print("The model info of {} is not profiled yet".format(model_name))
        sys.exit()


def load_scheduler_file(filename):
    abs_path = os.path.abspath(filename)
    if os.path.exists(abs_path):
        with open(filename, 'rb') as f:
            tensor_names = pickle.load(f)
            options = pickle.load(f)
            tensor_sizes = pickle.load(f)
        return tensor_names, options, tensor_sizes
    else:
        print("No scheduler file {} is found. Use FP32 for all tensors!".format(abs_path))
        return None, None, None

    
def json_load_scheduler_file(filename):
    abs_path = os.path.abspath(filename)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    with open(abs_path, 'r') as json_file:
        data = json.load(json_file)
    return data


"""
usage example: 
   python simulator.py --model=vgg16 --nvlink --cpu --compressor randomk
"""
def main():
    args = parse_args()
    
    if args.read_json != "":
        print(args.read_json)
        print(json_load_scheduler_file(args.read_json))
        return

    model_name = args.model
    simulator = init_simulator(args)
    simulator.get_stat()

    comm_device = "pcie" if not args.nvlink else "nvlink"
    comp_type = args.compressor
    two_level = "_two" if args.two_level else ""
    cpu = "_cpu" if args.cpu else ""
    nodes = "_" + str(args.nodes) + "nodes"

    base_overhead = simulator.get_base_overhead()
    options = simulator.set_model_options()
    gpu_overhead = simulator.get_total_overhead()
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    filename = model_name+ "/" + comm_device + "_" + comp_type + two_level + cpu + ".json"
    options_gpu = simulator.set_final_comm(copy.deepcopy(options))
    simulator.json_dump_options(filename, options_gpu)
    data = simulator.json_load_options(filename)
    # update_bert_options = [0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 5, 0, 5, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4]
    # simulator.update_options(filename, update_options)
    # options_gpu = [0, 1, 1, 0, 0, 1, 1, 0, 0, 9]
    # simulator.update_options(filename, options_gpu)
    print(filename, data)


if __name__ == "__main__":
    main()