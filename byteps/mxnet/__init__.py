# Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
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

from byteps.mxnet.ops import push_gradients, pull_gradients, broadcast_
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
                push_gradients(grad[i], average=True, name=str(index[i]))
                pull_gradients(grad[i], average=True, name=str(index[i]))
        else:
            push_gradients(grad, average=True, name=str(index))
            pull_gradients(grad, average=True, name=str(index))

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
def _append_broadcast_init(param, root_rank):
    init_impl = getattr(param, '_init_impl')
    def wrapped_init_impl(self, *args, **kwargs):
        init_impl(*args, **kwargs)
        broadcast_(self.data(), root_rank=root_rank)
        self.data().wait_to_read()
    return wrapped_init_impl


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
    return
