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
# =============================================================================
"""Inter-process communication using BytePS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import ctypes
from enum import Enum
import random
import string

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

from byteps.common import get_ext_suffix
from byteps.common import BytePSBasics as _BytePSBasics
from byteps.tensorflow.util import _executing_eagerly
import tensorflow as tf


def _load_library(name):
    """Loads a .so file containing the specified operators.
    Args:
      name: The name of the .so file to load.
    Raises:
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    return library


C_LIB = _load_library('c_lib' + get_ext_suffix())

_basics = _BytePSBasics(__file__, 'c_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
suspend = _basics.suspend
resume = _basics.resume
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank

dll_path = os.path.join(os.path.dirname(__file__),
                        'c_lib' + get_ext_suffix())
TF_LIB_CTYPES = ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)

def get_average_backwards_compatibility_fun(reduce_ops):
    """
    Handle backwards compatibility between the old average and the new op parameters.
    Old code using the average parameter (e.g. bps.PushPull(tensor, average=False))
    gets unchanged behavior, but mixing old and new is disallowed (e.g. no
    bps.PushPull(tensor, average=False, op=bps.Adasum)).
    """
    def impl(op, average):
        if op != None:
            if average != None:
                raise ValueError('The op parameter supersedes average. Please provide only one of them.')
            return op
        elif average != None:
            warnings.warn('Parameter `average` has been replaced with `op` and will be removed',
                          DeprecationWarning)
            return reduce_ops.Average if average else reduce_ops.Sum
        else:
            return reduce_ops.Average
    return impl

class ReduceOps(Enum):
    # This value should never appear past framework code, as
    # averaging is taken care of there.
    Average = "Average"
    Sum = "Sum"
    Adsum = "Adsum"

handle_average_backwards_compatibility = get_average_backwards_compatibility_fun(ReduceOps)


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)

def randomString(stringLength=16):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def _push_pull(tensor, scope='', name=None):
    """An op which sums an input tensor over all the BytePS processes.
    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all BytePS processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.
    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
    """
    if name is None and not _executing_eagerly():
        name = 'BytePSPushPull_%s' % _normalize_name(tensor.name)
    if scope == '' and not _executing_eagerly():
        if 'v1' in dir(tf.compat):
            scope = tf.compat.v1.get_default_graph().get_name_scope()
        else:
            scope = tf.get_default_graph().get_name_scope()
        if scope != '':
            scope += '/'
    if not name:
        name = ''
    full_name = scope + name
    if not full_name:
        full_name = "empty_name_" + randomString()
    full_name_ascii = full_name.encode("ascii")
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor(ctypes.c_char_p(full_name_ascii))
    return C_LIB.byteps_push_pull(tensor, name=name, input_name = full_name)


@ops.RegisterGradient('BytePSPushPull')
def _push_pull_grad(op, grad):
    """Gradient for push_pull op.
    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.
    Returns:
      The gradient with respect to the input of the op.
    """
    return _push_pull(grad)


def broadcast(tensor, root_rank, scope='', name=None, is_variable=True):
    """An op which broadcasts the input tensor on root rank to the same input tensor
    on all other BytePS processes.
    The broadcast operation is keyed by the name of the op. The tensor type and
    shape must be the same on all BytePS processes for a given name. The broadcast
    will not start until all processes are ready to send and receive the tensor.
    Returns:
      A tensor of the same shape and type as `tensor`, with the value broadcasted
      from root rank.
    """
    # Broadcast is implemented as push + pull after zero-ing non-root tensors
    if name is None and not _executing_eagerly():
        name = 'BytePSBroadcast_%s' % _normalize_name(tensor.name)
    if scope == '' and not _executing_eagerly():
        if 'v1' in dir(tf.compat):
            scope = tf.compat.v1.get_default_graph().get_name_scope()
        else:
            scope = tf.get_default_graph().get_name_scope()
        if scope != '':
            scope += '/'
    if not name:
        name = ''
    full_name = scope + name
    if not full_name:
        full_name = "empty_name_" + randomString()
    full_name_ascii = full_name.encode("ascii")

    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor(ctypes.c_char_p(full_name_ascii))
    if root_rank != rank():
        if is_variable:
            if hasattr(tf, 'assign_sub'):
                with tf.control_dependencies([tf.assign_sub(tensor, tensor)]):
                    return C_LIB.byteps_push_pull(tensor, name=name)
            else:
                with tf.control_dependencies([tf.compat.v1.assign_sub(tensor, tensor)]):
                    return C_LIB.byteps_push_pull(tensor, name=name, input_name = full_name)
        else:
            with tf.device(tensor.device):
                input_tensor = tf.zeros_like(tensor)
            return C_LIB.byteps_push_pull(input_tensor, name=name, input_name = full_name)
    else:
        return C_LIB.byteps_push_pull(tensor, name=name, input_name = full_name)


@ops.RegisterGradient('BytePSBroadcast')
def _broadcast_grad(op, grad):
    """Gradient for broadcast op.
    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.
    Returns:
      The gradient with respect to the input of the op.
    """
    root_rank = op.get_attr('root_rank')
    grad_reduced = _push_pull(grad)
    if rank() != root_rank:
        return grad_reduced * 0
    return grad_reduced
