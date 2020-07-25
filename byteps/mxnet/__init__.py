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

from __future__ import absolute_import, division, print_function

import copy
import os
import struct
import warnings

import mxnet as mx
import mxnet.ndarray as nd

from byteps.mxnet.compression import Compression
from byteps.mxnet.ops import (byteps_declare_tensor, byteps_push_pull, init,
                              local_rank, local_size, rank, resume, shutdown,
                              size, suspend)

parameter_index = 0


class DistributedOptimizer(mx.optimizer.Optimizer):
    """This is where BytePS's DistributedOptimizer wrapper for MXNet goes"""

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._enable_async = (int(os.getenv('BYTEPS_ENABLE_ASYNC', 0)) != 0)
        if self._enable_async:
            assert int(os.getenv('DMLC_NUM_WORKER')) > 1, \
                "Async is only valid for distributed training"
            print('BytePS: enable asynchronous training')

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_push_pull(self, index, grad):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                byteps_declare_tensor("gradient_" + str(index[i]))
                byteps_push_pull(grad[i], version=0, priority=-index[i],
                                 name="gradient_" + str(index[i]), is_average=True)
        else:
            byteps_declare_tensor("gradient_" + str(index))
            byteps_push_pull(grad, version=0, priority=-index,
                             name="gradient_" + str(index), is_average=True)

    def _do_push_pull_param(self, index, delta_weight):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                byteps_declare_tensor("weight_" + str(index[i]))
                byteps_push_pull(delta_weight[i], version=0, priority=-index[i],
                                 name="weight_" + str(index[i]), is_average=False)
        else:
            byteps_declare_tensor("weight_" + str(index))
            byteps_push_pull(delta_weight, version=0, priority=-index,
                             name="weight_" + str(index), is_average=False)

    def update(self, index, weight, grad, state):
        if self._enable_async:
            # create a tmp list for storing the original weight
            temp_weight_list = [w.copy() for w in weight]
            assert len(temp_weight_list) == len(weight)

            # update parameter locally
            self._optimizer.update(index, weight, grad, state)

            # get delta weight
            for i, temp_weight in enumerate(temp_weight_list):
                weight[i].__isub__(temp_weight)

            # push delta weight, and pull weight back to the same tensor
            self._do_push_pull_param(index, weight)

        else:
            self._do_push_pull(index, grad)
            self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        if self._enable_async:
            # create a tmp list for storing the original weight
            temp_weight_list = [w.copy() for w in weight]
            assert len(temp_weight_list) == len(weight)

            # update parameter locally
            self._optimizer.update_multi_precision(index, weight, grad, state)

            # get delta weight
            for i, temp_weight in enumerate(temp_weight_list):
                weight[i].__isub__(temp_weight)

            # push delta weight, and pull weight back to the same tensor
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
            byteps_declare_tensor("parameter_" + str(parameter_index))
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
        constructor for a list of additional supported arguments
    root_rank : int
        rank of root
    compression_params : dict
        Key-word arguments to be passed to gradient compression constructor. For example, 
        `{'compressor': 'onebit', 'ef': 'vanilla', 'momentum': 'nesterov', 'scaling': true}`.
        All compressor accept 'compressor', 'ef'. See each compressor's constructor for a list 
        of additional supported arguments
    """

    def __init__(self, params, optimizer, optimizer_params=None, root_rank=0, compression_params=None):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        param_list = []
        if isinstance(params, mx.gluon.ParameterDict):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])

        self._intra_compressor = self._register_compressor(
            params, optimizer_params, compression_params)

        super(DistributedTrainer, self).__init__(
            param_list, optimizer, optimizer_params=optimizer_params, kvstore=None)

        if local_rank() == 0:
            self._f = open("lr.s", "wb")
            self._f.truncate(8)

        self._bps_size = size()
        self.root_rank = root_rank
        self._intra_compressors = {}
        for i, param in enumerate(self._params):
            byteps_declare_tensor("parameter_" + str(i))
            self._intra_compressors[param.name] = type(self._intra_compressor)(
                **self._intra_compressor.__dict__)
            if param.grad_req != 'null':
                byteps_params = dict(
                    filter(lambda attr: attr[0].startswith(
                        "byteps_",), param.__dict__.items())
                )
                byteps_declare_tensor("gradient_" + str(i), **byteps_params)

    def _register_compressor(self, params, optimizer_params, compression_params):
        """Register compressor for BytePS

        params : mx.gluon.ParameterDict 
        optimizer_params : dict
        compression_params : dict
        """
        intra_compressor = Compression.none
        if not compression_params:
            return intra_compressor

        if compression_params.get("fp16"):
            intra_compressor = Compression.fp16

        if "compressor" not in compression_params:
            warnings.warn("Compressor is not defined")
            return intra_compressor

        check_list = ["compressor", "ef", "momentum"]

        for _, param in params.items():
            # generic
            for item in check_list:
                if compression_params.get(item):
                    if isinstance(compression_params[item], str):
                        setattr(param, "byteps_%s_type" %
                                item, compression_params[item])
                    else:
                        raise TypeError("%s should be str" % item)

            # need parameter
            compressor = compression_params["compressor"]
            if compressor == "onebit":
                setattr(param, "byteps_compressor_onebit_scaling", str(
                    compression_params.get("scaling", False)))
            elif compressor == "topk" or compressor == "randomk" or compressor == "dithering":
                # raise KeyError if 'k' is not found
                setattr(param, "byteps_compressor_k",
                        compression_params["k"])

            if compression_params.get("momentum"):
                setattr(param, "byteps_momentum_mu",
                        optimizer_params["momentum"])

            if compression_params.get("seed", None) is not None:
                setattr(param, "byteps_seed",
                        compression_params["seed"])

        # the following code will delete some items in `optimizer_params`
        # to avoid duplication
        if compression_params.get("momentum"):
            # 1bit compressor use an additional momentum for weight decay
            if compressor == "onebit" and "wd" in optimizer_params:
                intra_compressor = Compression.wdmom(
                    intra_compressor, optimizer_params["momentum"], optimizer_params["wd"])
                del optimizer_params["wd"]

            del optimizer_params['momentum']

        return intra_compressor

    def step(self, batch_size, ignore_stale_grad=False):
        # grad is normalized with batch_size. setting _scale to batch_size is
        # to prevent normalized by batch_size twice.
        self._scale = batch_size
        super(DistributedTrainer, self).step(batch_size, ignore_stale_grad)

    def _allreduce_grads(self):
        # update lr
        if local_rank() == 0:
            self._f.seek(0)
            ba = struct.pack("d", self.learning_rate)
            self._f.write(ba)
            self._f.flush()

        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                # normalized with batch_size and num_workers
                nd._internal._mul_scalar(
                    param._grad[0], 1.0 / self._scale / self._bps_size, out=param._grad[0])
                compressed, ctx = self._intra_compressors[param.name].compress(
                    param._grad[0])
                byteps_push_pull(compressed, is_average=False,
                                 name="gradient_" + str(i), priority=-i)
                param._grad[0][:] = self._intra_compressors[param.name].decompress(
                    compressed, ctx,  x=param._data[0])

    def _init_params(self):
        tensors = []
        for param in self._params_to_init:
            if param._deferred_init:
                tensors.append(param)
            else:
                param_arrays = param._check_and_get(param._data, list)
                idx = self._param2idx[param.name]

                if rank() != self.root_rank:
                    param_arrays[0].__imul__(0)

                compressed, ctx = self._intra_compressors[param.name].compress(
                    param_arrays[0])
                byteps_push_pull(compressed, version=0, priority=0,
                                 name="parameter_" + str(idx), is_average=False)
                param_arrays[0][:] = self._intra_compressors[param.name].decompress(
                    compressed, ctx,  x=param._data[0])

        self._params_to_init = tensors
