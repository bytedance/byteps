# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import mxnet as mx
import os

from byteps.mxnet.ops import byteps_push_pull, byteps_declare_tensor
from byteps.mxnet.ops import init, shutdown
from byteps.mxnet.ops import size, local_size, rank, local_rank

# huhanpeng
from byteps.mxnet.ops import get_comm_time
import logging
import sys, os
from mxnet import profiler
import json
import networkx as nx

parameter_index = 0

# huhanpeng:
def log(s):
    if rank() == 0:
        print(s)
        sys.stdout.flush()

class Recorder(object):
    # huhanpeng: class used to collect trace info
    def __init__(self, only_symbolic=True):
        self.time_dict = {"traceEvents":[]}
        self.idx_dict = {}
        self.gradient_name_list = None
        self.step_cnt = 0
        if os.environ.get("TRACE_ON", "") != 'ON':
            self._end_trace = True
            return
        self._end_trace = False
        self.end_step = int(os.environ.get("TRACE_END_STEP", "10"))
        self.trace_dir = os.environ.get("TRACE_DIR", ".") + "/" + os.environ.get("BYTEPS_RANK") + "_" + os.environ.get("BYTEPS_LOCAL_RANK") + "/"
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)
        self.trace_path = self.trace_dir + 'bps_trace_local_rank%s_%dstep.json' % (os.environ.get("BYTEPS_LOCAL_RANK"), self.end_step)

        """config the mxnet profile"""
        if only_symbolic:
            profiler.set_config(profile_symbolic=True,
                        profile_imperative=False,
                        profile_memory=False,
                        profile_api=False,
                        # profile_process=False,
                        aggregate_stats=False, 
                        filename=self.trace_dir+'temp.json')
        else:
            profiler.set_config(profile_all=True, 
                        aggregate_stats=False, 
                        filename=self.trace_dir+'temp.json')
        profiler.set_state('run')
        self.dag = nx.DiGraph()

        #! symbol/block, used to get the dependency info, at least one should be given
        self.block = None
        self.symbol = None

    def scheduler(self, index, _check_stop=False):
        '''A scheduler, manage the counter for each gradient, `self.idx_dict` is 
        used to record the status of each gradient, the fist time a gradinet call 
        this function, register the `index` to self.idx_dict with False; when it
        becomes True, this gradinet is ready to output traces (the communication 
        traces of this gradient have been collected); Output traces only when 
        the status of gradients are True.

        Parameters:
        ----------
        index : int
            The index of the gradient.
        _check_stop : bool
            if the flag is set, add the step_cnt by 1.

        Returns:
        ----------
        bool, whether to collect the communication trace of this gradinet.
        '''
        if self._end_trace:
            return False

        if index not in self.idx_dict:
            self.idx_dict[index] = False

        if self.idx_dict[index]:
            if False not in self.idx_dict.values():
                """All parameters have been recorded, end profiling"""
                self._end_trace = True   
                self.save_trace()
            return False # the communication traces of this parameter have been read

        """ Since each parameter will call this function, to decide when to stop profiling,
            we only focus on one parameter, e.g., the first parameter.
        """
        if _check_stop:
            self.step_cnt += 1
            
        if self.step_cnt >= self.end_step:
            if self.gradient_name_list is None:
                self.gradient_name_list = []
                with open(os.path.join(os.environ.get("TRACE_DIR", ".") + "/", 'arg_namesINpara_names.txt'), 'r') as lines:
                    for line in lines:
                        name = line[:-1]
                        self.gradient_name_list.append(name)
            return True
        else:
            return False            

    def end_trace(self):
        return self._end_trace

    def save_trace(self):
        ''' Output trace resutls '''
        # -- Output mxnet traces and import it
        profiler.set_state('stop')
        profiler.dump()
        with open(self.trace_dir + 'temp.json', 'r') as f:
            mxnet_traces = json.load(f)

        # -- Get the dependency graph, adapt to DistributedOptimizer and DistributedTrainer
        if self.symbol is not None:
            self.gen_dag(self.symbol.debug_str())
        elif self.block is not None:
            symbol = self.block._cached_graph[1]
            self.gen_dag(symbol.debug_str())
        else:
            raise ValueError("A symbol or model/block must be given when defining DistributedOptimizer/DistributedTrainer.")

        # -- Apply dependencies in self.dag to the mxnet traces.
        rst_traces = self.add_dependency(mxnet_traces)

        # -- Combine two kinds of trace and output them
        self.time_dict["traceEvents"] += rst_traces["traceEvents"]
        with open(self.trace_path, 'w') as f:
            json.dump(self.time_dict, f, indent=4)

        # -- Output the dag, only containing forward info
        nx.write_gml(self.dag, self.trace_dir + "dag.gml", lambda x: str(x))
        log("Stop tracing, output trace: %s" % self.trace_path)
        # -- clear the time dict after save it
        self.time_dict = None

    def add_dependency(self, mxnet_traces):
        '''Apply dependency info to the mxnet trace results

        Parameters
        ----------
        mxnet_traces : dict
            A dict containing MXNet trace results.

        Returns
        ----------
        rst_traces : dict
            A dict containing MXNet trace results combined with dependency info.
        '''
        index = 0
        rst_traces = {"traceEvents": []}
        while index < len(mxnet_traces["traceEvents"]):
            trace = mxnet_traces["traceEvents"][index]
            name = trace["name"]
            # -- add for mxnet-gluon case
            if "name=" in name:
                name = name.split("name=")[1].split(";")[0]

            if trace["ph"] != 'B' and trace["ph"] != 'b':
                index += 1
                continue
            if "_backward" in name: # backward nodes
                name = name.split("_backward")[0]
                if "_fwd" in name: # -- for mxnet-gluon case
                    name = name.split("_fwd")[0]
                if name not in self.dag.nodes:
                    index += 1
                    continue    
                innodes = ["BW." + _n for _n in self.dag.successors(name)]
                name = "BW." + name
            else:
                if "_fwd" in name: # -- for mxnet-gluon case
                    name = name.split("_fwd")[0]
                if name not in self.dag.nodes:
                    index += 1
                    continue
                else:
                    # forward nodes
                    innodes = ["FW." + _n for _n, _ in self.dag.in_edges(name)] + self.dag.nodes[name]["var"]
                    name = "FW." + name
            args = {"name": name}
            for i, _n in enumerate(innodes):
                args["arg%d"%i] = _n
            trace["name"] = name
            trace["args"] = args
            trace["ph"] = "X"

            # -- each 'B/b' type event is followed with a 'E/e' event, skip them
            while True: 
                index += 1
                next_trace = mxnet_traces["traceEvents"][index]
                if next_trace["ph"] == 'e' or next_trace["ph"] == 'E':
                    break
            if name.split(".")[1] not in next_trace["name"]:
                raise ValueError("'b/B' events must be followed with 'e/E' events!!!")
            trace["dur"] = next_trace['ts'] - trace['ts']
            rst_traces["traceEvents"].append(trace)
            index += 1

        return rst_traces

    def gen_dag(self, s):
        """Construct a DAG from the mxnet info

        Parameters:
        ----------
        s : str
            Must follow the standard chrome trace format and not None.
        """
        blocks = s.split("--------------------\n")
        index = 0
        for i in range(1, len(blocks)):
            prev_block = blocks[i-1]
            var = []
            prev_ls = prev_block.split('\n')
            for l in prev_ls:
                if "Variable" in l:
                    var.append(l.split('Variable:')[1])
            block = blocks[i]
            ls = block.split('\n')
            if 'Name' not in ls[0]:
                index += 1
                continue
            name = ls[0].split('Name=')[1]
            args = []
            for l in ls:
                if "arg[" in l:
                    arg_name = l.split(']=')[1].split('(')[0]
                    if arg_name not in var:
                        args.append(arg_name)
            if "_fwd" in name:
                name = name.split("_fwd")[0]
            for innode in args:
                innode = innode.split("_fwd")[0] if "_fwd" in innode else innode
                self.dag.add_edges_from([(innode, name)])
            if name in self.dag.nodes:
                self.dag.nodes[name]["var"] = ["Comm." + e for e in var]
            else:
                # for the first node, it has no arg, so not be defined yet
                self.dag.add_node(name, var=["Comm." + e for e in var])           
            index += 1

    def byteps_collect_comm(self, index, tensor, name):
        ''' Offline collect the communication trace results of gradient `index`

        Parameters
        ----------
        index : int
            The index of the gradient.
        tensor: tensor
            A tensor to average and sum.
        name : str
            A name of the reduction operation.
        '''
        # huhanpeng: can be removed
        if self.end_trace():
            return

        # -- read communication traces offline
        _ts_dur_list = get_comm_time(tensor, name) 

        def return_event(index, _ts, _dur):
            if _ts == 0:
                raise ValueError("_ts should not be 0")
            para_name = self.gradient_name_list[index]
            op_name = "_".join(para_name.split("_")[:-1])
            return {
                    "name": "Comm." + para_name,
                    "ts": _ts,
                    "dur": _dur,
                    "ph": "X",
                    "pid": "Comm." + para_name,
                    "args": {
                        "name": "Comm." + para_name,
                        "input0": "BW." + op_name
                        }
                    }
        self.time_dict["traceEvents"] += [return_event(index, _ts, _dur) for (_ts, _dur) in _ts_dur_list]
        self.idx_dict[index] = True # avoid repeatedly read
        # log("_ts: %s, _dur: %s" % (str(_ts), str(_dur)))

class DistributedOptimizer(mx.optimizer.Optimizer):
    """This is where BytePS's DistributedOptimizer wrapper for MXNet goes"""
    def __init__(self, optimizer, sym=None):
        self._optimizer = optimizer
        # huhanpeng: debug
        log("This is a new DistributedOptimizer with auto profiling")
        """tracing configure""" 
        self.recorder = Recorder()
        self.recorder.symbol = sym

        self._enable_async = (int(os.getenv('BYTEPS_ENABLE_ASYNC', 0)) != 0)
        if self._enable_async:
            assert int(os.getenv('DMLC_NUM_WORKER'))>1, \
                "Async is only valid for distributed training"
            print('BytePS: enable asynchronous training')

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_push_pull(self, index, grad):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                byteps_declare_tensor(grad[i], "gradient_" + str(index[i]))
                byteps_push_pull(grad[i], version=0, priority=-index[i],
                                 name="gradient_" + str(index[i]), is_average=True)
        else:
            byteps_declare_tensor(grad, "gradient_" + str(index))
            byteps_push_pull(grad, version=0, priority=-index,
                             name="gradient_" + str(index), is_average=True)
        # huhanpeng: modify scheduler for when the index is tuple or list, 
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                if self.recorder.scheduler(index[i], (True if index[i] == 0 else False)):
                    self.recorder.byteps_collect_comm(index[i], grad[i], "gradient_" + str(index[i]))       
        else:
            if self.recorder.scheduler(index, (True if index == 0 else False)):
                self.recorder.byteps_collect_comm(index, grad, "gradient_" + str(index))


    def _do_push_pull_param(self, index, delta_weight):
        # huhanpeng: not implemented
        raise ValueError("Not implemented")

        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                byteps_declare_tensor(delta_weight[i], "weight_" + str(index[i]))
                byteps_push_pull(delta_weight[i], version=0, priority=-index[i],
                                 name="weight_" + str(index[i]), is_average=False)
        else:
            byteps_declare_tensor(delta_weight, "weight_" + str(index))
            byteps_push_pull(delta_weight, version=0, priority=-index,
                             name="weight_" + str(index), is_average=False)

    def update(self, index, weight, grad, state):
        if self._enable_async:
            temp_weight = weight.copy()
            self._optimizer.update(index, weight, grad, state)
            # push delta weight, and pull weight back to the same tensor
            weight.__isub__(temp_weight)
            self._do_push_pull_param(index, weight)
        else:
            self._do_push_pull(index, grad)
            self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        if self._enable_async:
            temp_weight = weight.copy()
            self._optimizer.update_multi_precision(index, weight, grad, state)
            # push delta weight, and pull weight back to the same tensor
            weight.__isub__(temp_weight)
            self._do_push_pull_param(index, weight)
        else:
            self._do_push_pull(index, grad)
            self._optimizer.update_multi_precision(index, weight, grad, state)

    def set_learning_rate(self, lr):
        self._optimizer.set_learning_rate(lr)

    def set_lr_mult(self, args_lr_mult):
        self._optimizer.set_lr_mult(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        self._optimizer.set_wd_mult(args_wd_mult)


def broadcast_parameters(params, root_rank=0):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `Module.get_params()`.

    Arguments:
        params: dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    global parameter_index

    if isinstance(params, dict):
        tensors = [p for _, p in sorted(params.items())]

        # Run tensor initilization
        for i in range(len(tensors)):
            byteps_declare_tensor(tensors[i], "parameter_" + str(parameter_index))
            # Broadcast is implemented as push + pull in BytePS
            # To broadcast: we should zero-out all non-root tensors, and disable push_pull average
            if rank() != root_rank:
                tensors[i].__imul__(0)
            byteps_push_pull(tensors[i], version=0, priority=0,
                             name="parameter_" + str(parameter_index), is_average=False)
            parameter_index += 1

        # Make sure tensors pushed to MXNet engine get processed such that all
        # workers are synced before starting training.
        for tensor in tensors:
            tensor.wait_to_read()

    elif isinstance(params, mx.gluon.parameter.ParameterDict):
        raise TypeError("For gluon users, you should not call this function. "
                        "DistributedTrainer will broadcast all parameters at "
                        "the first training step.")

    else:
        raise ValueError('Invalid params of type: %s' % type(params))


class DistributedTrainer(mx.gluon.Trainer):
    """A subclass of MXNet gluon.Trainer.

    There are two differences between DistributedTrainer and Trainer:
    1. DistributedTrainer calculates gradients using BytePS push pull
       API while Trainer does it using kvstore push/pull APIs;
    2. DistributedTrainer performs push_pull(summation) and average,
       while Trainer only performs push_pull(summation).

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    """

    def __init__(self, params, optimizer, 
                optimizer_params=None, 
                root_rank=0, 
                block=None,
                batch_data=None,
                ctx=None,
                data_num=None):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        # huhanpeng: debug
        log("This is a new DistributedTrainer with auto profiling")
        self.recorder = Recorder(only_symbolic=False)
        # self.recorder.gradient_name_list = [param.name for param in list(params.values)]
        self.recorder.gradient_name_list = [gradient_name for gradient_name in list(params)]
        if block is None:
            raise ValueError("`block` must be given to define DistributedTrainer")
        self.recorder.block = block
        self.imported_net = None

        if not self.recorder.end_trace():
            if batch_data is None or ctx is None:
                raise ValueError("`batch_data` and `ctx` must be given if you want to call auto-profiling.")

            # --------------------- warmup and export ---------------
            output = block(*batch_data)
            prefix = "GluonModel"
            block.export(prefix)
            assert os.path.isfile(prefix + '-symbol.json')
            assert os.path.isfile(prefix + '-0000.params')

            # --------------------- import with SymbolBlock ----------
            if data_num:
                _data = ['data%d'%i for i in range(data_num)]
            else:
                _data = ['data']
            self.imported_net = mx.gluon.nn.SymbolBlock.imports(prefix + '-symbol.json',
                                                               _data,
                                                               prefix + '-0000.params',
                                                               ctx=ctx)
            self.imported_net.hybridize(static_shape=True, static_alloc=True)
            # BytePS: fetch and broadcast parameters
            params = self.imported_net.collect_params()


        super(DistributedTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by BytePS size, which is equivalent to performing
        # average in push_pull, has better performance.
        self._scale /= size()
        self.root_rank = root_rank

    def update_model(self):
        if self.recorder.end_trace():
            return self.recorder.block
        else:
            return self.imported_net

    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                byteps_declare_tensor(param.list_grad()[0], "gradient_" + str(i))
                byteps_push_pull(param.list_grad()[0], is_average=False,
                                 name="gradient_" + str(i), priority=-i)
            # huhanpeng
            if self.recorder.scheduler(i, (True if i == 0 else False)) and param.grad_req != 'null':
                self.recorder.byteps_collect_comm(i, param.list_grad()[0], "gradient_" + str(i))

    def _init_params(self):
        tensors = []
        for param in self._params_to_init:
            if param._deferred_init:
                tensors.append(param)
            else:
                param_arrays = param._check_and_get(param._data, list)
                idx = self._param2idx[param.name]
                byteps_declare_tensor(param_arrays[0], "parameter_" + str(idx))

                if rank() != self.root_rank:
                    param_arrays[0].__imul__(0)
                byteps_push_pull(param_arrays[0], version=0, priority=0,
                                 name="parameter_" + str(idx), is_average=False)
                param_arrays[0].wait_to_read()

        self._params_to_init = tensors
