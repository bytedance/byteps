# Copyright 2019 Bytedance Inc. All Rights Reserved.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
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
# pylint: disable=g-short-docstring-punctuation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from byteps.tensorflow.compression import Compression
from byteps.tensorflow.ops import broadcast, _push_pull
from byteps.tensorflow.ops import init, shutdown, suspend, resume
from byteps.tensorflow.ops import size, local_size, rank, local_rank
from byteps.tensorflow.util import _executing_eagerly

import tensorflow as tf
import os, sys
from tensorflow.python.ops import control_flow_ops

def push_pull(tensor, scope='', average=True, device_dense='', device_sparse='',
              compression=Compression.none, enable_async=False):
    """Perform an push_pull on a tf.Tensor or tf.IndexedSlices.
    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
                The shape of the input must be identical across all ranks.
        scope: the graph name scope
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.
        device_dense: Device to be used for dense tensors. Uses GPU by default.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
    Returns:
        A tensor of the same shape and type as `tensor`, summed across all
        processes.
    """
    with tf.device(device_dense):
        byteps_size = tf.cast(size(), dtype=tensor.dtype)
        tensor_compressed, ctx = compression.compress(tensor)
        summed_tensor_compressed = _push_pull(tensor_compressed, scope)
        summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
        if not enable_async:
            _div = tf.div if hasattr(tf, 'div') else tf.math.divide
            new_tensor = (_div(summed_tensor, byteps_size)
                          if average else summed_tensor)
        else: # no need to average for async training
            new_tensor = summed_tensor
    return new_tensor


def broadcast_global_variables(root_rank, scope=''):
    """Broadcasts all global variables from root rank to all other processes.
    Arguments:
        root_rank: rank of the process from which global variables will be broadcasted
        to all other processes.
        scope: the graph name scope
    """
    return broadcast_variables(tf.global_variables(), root_rank, scope)


def broadcast_variables(variables, root_rank, scope=''):
    """Broadcasts variables from root rank to all other processes.
    Arguments:
        variables: variables for broadcast
        root_rank: rank of the process from which global variables will be broadcasted
                   to all other processes.
        scope: the graph name scope
    """
    _assign = tf.assign if hasattr(tf, 'assign') else tf.compat.v1.assign
    return tf.group(*[_assign(var, broadcast(var, root_rank, scope))
                      for var in variables])

try:
    _get_default_graph = tf.get_default_graph
except AttributeError:
    try:
        _get_default_graph = tf.compat.v1.get_default_graph
    except AttributeError:
        _get_default_graph = None

try:
    _SessionRunHook = tf.estimator.SessionRunHook
except AttributeError:
    try:
        _SessionRunHook = tf.train.SessionRunHook
    except AttributeError:
        _SessionRunHook = None

if _SessionRunHook is not None and _get_default_graph is not None:
    class BroadcastGlobalVariablesHook(_SessionRunHook):
        """
        SessionRunHook that will broadcast all global variables from root rank
        to all other processes during initialization.
        This is necessary to ensure consistent initialization of all workers when
        training is started with random weights or restored from a checkpoint.
        """

        def __init__(self, root_rank, device=''):
            """Construct a new BroadcastGlobalVariablesHook that will broadcast all
            global variables from root rank to all other processes during initialization.
            Args:
            root_rank:
                Rank that will send data, other ranks will receive data.
            device:
                Device to be used for broadcasting. Uses GPU by default
                if BytePS was build with BYTEPS_GPU_BROADCAST.
            """
            super(BroadcastGlobalVariablesHook, self).__init__()
            self.root_rank = root_rank
            self.bcast_op = None
            self.device = device

        def begin(self):
            if not self.bcast_op or self.bcast_op.graph != _get_default_graph():
                with tf.device(self.device):
                    self.bcast_op = broadcast_global_variables(self.root_rank)

        def after_create_session(self, session, coord):
            session.run(self.bcast_op)

try:
    # TensorFlow 2.x
    _LegacyOptimizer = tf.compat.v1.train.Optimizer
except AttributeError:
    try:
        # TensorFlow 1.x
        _LegacyOptimizer = tf.train.Optimizer
    except AttributeError:
        # Future TensorFlow versions
        _LegacyOptimizer = None

if _LegacyOptimizer is not None:
    class DistributedOptimizer(_LegacyOptimizer):
        """An optimizer that wraps another tf.Optimizer, using an push_pull to
        average gradient values before applying gradients to model weights."""

        def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                    device_sparse='', compression=Compression.none,
                    sparse_as_dense=False):
            """Construct a new DistributedOptimizer, which uses another optimizer
            under the hood for computing single-process gradient values and
            applying gradient updates after the gradient values have been averaged
            across all the BytePS ranks.
            Args:
            optimizer:
                Optimizer to use for computing gradients and applying updates.
            name:
                Optional name prefix for the operations created when applying
                gradients. Defaults to "Distributed" followed by the provided
                optimizer type.
            use_locking:
                Whether to use locking when updating variables.
                See Optimizer.__init__ for more info.
            device_dense:
                Device to be used for dense tensors. Uses GPU by default.
            device_sparse:
                Device to be used for sparse tensors. Uses GPU by default.
            compression:
                Compression algorithm used during push_pull to reduce the amount
                of data sent during the each parameter update step.  Defaults to
                not using compression.
            sparse_as_dense:
                Treat all sparse gradients as dense tensors.  This can help improve
                performance and memory utilization if the original sparse gradient
                has high density.  Defaults to false.
            """
            if name is None:
                name = "Distributed{}".format(type(optimizer).__name__)

            self._optimizer = optimizer
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense

            self._enable_async = (int(os.getenv('BYTEPS_ENABLE_ASYNC', 0)) != 0)
            if self._enable_async:
                assert int(os.getenv('DMLC_NUM_WORKER')) > 1, \
                    "Async is only valid for distributed training"
                print('BytePS: enable asynchronous training')

            def push_pull_grads(grads):
                with tf.name_scope(self._name + "_Push_Pull") as scope:
                    if self._sparse_as_dense:
                        grads = [tf.convert_to_tensor(grad)
                                if grad is not None and isinstance(grad, tf.IndexedSlices)
                                else grad for grad in grads]

                    return [push_pull(grad, scope,
                                    device_dense=self._device_dense,
                                    device_sparse=self._device_sparse,
                                    compression=self._compression,
                                    enable_async=self._enable_async)
                            if grad is not None else grad
                            for grad in grads]

            if _executing_eagerly():
                self._push_pull_grads = tf.contrib.eager.defun(push_pull_grads)
            else:
                self._push_pull_grads = push_pull_grads

            super(DistributedOptimizer, self).__init__(
                name=name, use_locking=use_locking)

        def compute_gradients(self, *args, **kwargs):
            """Compute gradients of all trainable variables.
            See Optimizer.compute_gradients() for more info.
            In DistributedOptimizer, compute_gradients() is overriden to also
            push_pull the gradients before returning them.
            """
            gradients = self._optimizer.compute_gradients(*args, **kwargs)
            if size() > 1 and not self._enable_async:
                grads, vars = zip(*gradients)
                avg_grads = self._push_pull_grads(grads)
                return list(zip(avg_grads, vars))
            else:
                return gradients

        def apply_gradients(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            if self._enable_async: # async training
                grads_and_vars = args[0]
                _, vars = zip(*grads_and_vars)
                old_tensors = []
                for var in vars:
                    old_tensors.append(tf.convert_to_tensor(var))
                apply_ops = self._optimizer.apply_gradients(*args, **kwargs)
                with tf.control_dependencies([apply_ops]):
                    # get the delta
                    for i, var in enumerate(vars):
                        old_tensors[i] = tf.subtract(var, old_tensors[i])

                    # reuse the _push_pul_grads(), but is transferring parameters
                    updated_tensors = self._push_pull_grads(old_tensors)

                    # copy the updated variable back
                    assign_op_list = []
                    for i, tensor in enumerate(updated_tensors):
                        assign_op_list.append(tf.assign(vars[i], tensor))

                return control_flow_ops.group(*assign_op_list)
            else:
                return self._optimizer.apply_gradients(*args, **kwargs)

        def get_slot(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot(*args, **kwargs)

        def get_slot_names(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot_names(*args, **kwargs)

        def variables(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.variables(*args, **kwargs)


if hasattr(tf, 'GradientTape'):
    class _DistributedGradientTape(tf.GradientTape):

        def __init__(self, tape, device_dense, device_sparse,
                     compression, sparse_as_dense, persistent=False, watch_accessed_variables=True):
            if hasattr(tape, '_watch_accessed_variables'):
                super(self.__class__, self).__init__(persistent, watch_accessed_variables)
            else:
                super(self.__class__, self).__init__(persistent)
            self._tape = tape
            self._persistent = persistent
            self._watch_accessed_variables = watch_accessed_variables
            self._name = "Distributed"
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense

            def push_pull_grads(grads):
                with tf.name_scope(self._name + "_Push_Pull") as scope:
                    if self._sparse_as_dense:
                        grads = [tf.convert_to_tensor(grad)
                                 if grad is not None and isinstance(grad, tf.IndexedSlices)
                                 else grad for grad in grads]
                    return [push_pull(grad, scope,
                                      device_dense=self._device_dense,
                                      device_sparse=self._device_sparse,
                                      compression=self._compression)
                            if grad is not None else grad
                            for grad in grads]

            self._push_pull_grads = push_pull_grads

        def gradient(self, target, sources, output_gradients=None):
            gradients = super(self.__class__, self).gradient(target, sources, output_gradients)
            if size() > 1:
                avg_grads = self._push_pull_grads(gradients)
                return avg_grads
            else:
                return gradients


    def DistributedGradientTape(gradtape, device_dense='', device_sparse='',
                                compression=Compression.none, sparse_as_dense=False):
        """An tape that wraps another tf.GradientTape, using an push_pull to
        average gradient values before applying gradients to model weights.
        Args:
        gradtape:
            GradientTape to use for computing gradients and applying updates.
        device_dense:
            Device to be used for dense tensors. Uses GPU by default.
        device_sparse:
            Device to be used for sparse tensors. Uses GPU by default.
        compression:
            Compression algorithm used during push_pull to reduce the amount
            of data sent during the each parameter update step.  Defaults to
            not using compression.
        sparse_as_dense:
            Treat all sparse gradients as dense tensors.  This can help improve
            performance and memory utilization if the original sparse gradient
            has high density.  Defaults to false.
        """
        cls = type(gradtape.__class__.__name__, (gradtape.__class__,),
                dict(_DistributedGradientTape.__dict__))
        if hasattr(gradtape, '_watch_accessed_variables'):
            return cls(gradtape._tape, device_dense, device_sparse,
                    compression, sparse_as_dense,
                    gradtape._persistent, gradtape._watch_accessed_variables)
        else:
            return cls(gradtape._tape, device_dense, device_sparse,
                    compression, sparse_as_dense,
                    gradtape._persistent)
