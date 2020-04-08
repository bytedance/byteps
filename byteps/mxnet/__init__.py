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
from byteps.mxnet.ops import init, shutdown, suspend, resume
from byteps.mxnet.ops import size, local_size, rank, local_rank

parameter_index = 0


class DistributedOptimizer(mx.optimizer.Optimizer):
    """This is where BytePS's DistributedOptimizer wrapper for MXNet goes"""
    def __init__(self, optimizer):
        self._optimizer = optimizer
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
        constructor for a list of additional supported arguments.
    """

    def __init__(self, params, optimizer, optimizer_params=None, root_rank=0):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        param_list = []
        if isinstance(params, mx.gluon.ParameterDict):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])

        super(DistributedTrainer, self).__init__(
            param_list, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by BytePS size, which is equivalent to performing
        # average in push_pull, has better performance.
        self._scale /= size()
        self.root_rank = root_rank
        for i, param in enumerate(self._params):
            byteps_declare_tensor("parameter_" + str(i))
            if param.grad_req != 'null':
                byteps_declare_tensor("gradient_" + str(i))


    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                byteps_push_pull(param.list_grad()[0], is_average=False,
                                 name="gradient_" + str(i), priority=-i)

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
                byteps_push_pull(param_arrays[0], version=0, priority=0,
                                 name="parameter_" + str(idx), is_average=False)

        self._params_to_init = tensors
