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

import threading
import warnings

from byteps.mxnet.ops import byteps_push_pull, byteps_declare_tensor
from byteps.mxnet.ops import init, shutdown
from byteps.mxnet.ops import size, local_size, rank, local_rank

import mxnet as mx
import types


# This is where BytePS's DistributedOptimizer wrapper for MXNet goes
class DistributedOptimizer(mx.optimizer.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_push_pull(self, index, grad):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                byteps_declare_tensor(grad[i], "gradient_"+str(index[i]))
                byteps_push_pull(grad[i], version=0, priority=-index[i], name="gradient_"+str(index[i]), is_average=True)
        else:
            byteps_declare_tensor(grad, "gradient_"+str(index))
            byteps_push_pull(grad, version=0, priority=-index, name="gradient_"+str(index), is_average=True)

    def update(self, index, weight, grad, state):
        self._do_push_pull(index, grad)
        self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        self._do_push_pull(index, grad)
        self._optimizer.update_multi_precision(index, weight, grad, state)

    def set_learning_rate(self, lr):
        self._optimizer.set_learning_rate(lr)

    def set_lr_mult(self, args_lr_mult):
        self._optimizer.set_lr_mult(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        self._optimizer.set_wd_mult(args_wd_mult)

# Wrapper to inject BytePS broadcast after parameter initialization
def _append_broadcast_init(param, root_rank, index):
    init_impl = getattr(param, '_init_impl')
    def wrapped_init_impl(self, *args, **kwargs):
        init_impl(*args, **kwargs)
        # Broadcast is implemented as push + pull in BytePS
        byteps_push_pull(self.data(), version=0, priority=0, name="parameter_"+str(index), is_average=False)
        self.data().wait_to_read()
    return wrapped_init_impl

parameter_index = 0

def broadcast_parameters(params, root_rank=0):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `Module.get_params()` or the
    `Block.collect_params()`.
    Arguments:
        params: One of the following:
            - dict of parameters to broadcast
            - ParameterDict to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    tensors = []
    if isinstance(params, dict):
        tensors = [p for _, p in sorted(params.items())]
    elif isinstance(params, mx.gluon.parameter.ParameterDict):
        for _, p in sorted(params.items()):
            try:
                tensors.append(p.data())
            except mx.gluon.parameter.DeferredInitializationError:
                # Inject wrapper method with post-initialization broadcast to
                # handle parameters with deferred initialization
                global parameter_index
                byteps_declare_tensor(p.data(), "parameter_"+str(parameter_index))
                new_init = _append_broadcast_init(p, root_rank, parameter_index)
                parameter_index += 1
                p._init_impl = types.MethodType(new_init, p)
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run tensor initilization
    for i in range(len(tensors)):
        byteps_declare_tensor(tensors[i], "parameter_"+str(parameter_index))
        # Broadcast is implemented as push + pull in BytePS
        # To broadcast: we should zero-out all non-root tensors, and disable push_pull average
        if rank() != root_rank:
            tensors[i].__imul__(0)
        byteps_push_pull(tensors[i], version=0, priority=0, name="parameter_"+str(parameter_index), is_average=False)
        parameter_index += 1

    # Make sure tensors pushed to MXNet engine get processed such that all
    # workers are synced before starting training.
    for tensor in tensors:
        tensor.wait_to_read()


class DistributedTrainer(mx.gluon.Trainer):
    """A subclass of MXNet gluon.Trainer.

    There are two differences between DistributedTrainer and Trainer:
    1. DistributedTrainer calculates gradients using Byteps push pull
       API while Trainer does it using kvstore push/pull APIs;
    2. DistributedTrainer performs allreduce(summation) and average
       while Trainer only performs allreduce(summation).

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

    def __init__(self, params, optimizer, optimizer_params=None):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        super(DistributedTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by byteps size, which is equivalent to performing
        # average in allreduce, has better performance.
        self._scale /= size()

    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                byteps_push_pull(param.list_grad()[0], is_average=False,
                                 name="parameter_"+str(i), priority=-i)

