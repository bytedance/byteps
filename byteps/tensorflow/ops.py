# Copyright 2019 ByteDance Inc. All Rights Reserved.
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

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

from byteps.common import get_ext_suffix
from byteps.common import BytePSBasics as _BytePSBasics
from byteps.tensorflow.util import _executing_eagerly


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
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank

dll_path = os.path.join(os.path.dirname(__file__),
                        'c_lib' + get_ext_suffix())
TF_LIB_CTYPES = ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)


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
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor(ctypes.c_char_p(scope+name))
    return C_LIB.byteps_push_pull(tensor, name=name)


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


def broadcast(tensor, root_rank, name=None):
    """An op which broadcasts the input tensor on root rank to the same input tensor
    on all other BytePS processes.
    The broadcast operation is keyed by the name of the op. The tensor type and
    shape must be the same on all BytePS processes for a given name. The broadcast
    will not start until all processes are ready to send and receive the tensor.
    Returns:
      A tensor of the same shape and type as `tensor`, with the value broadcasted
      from root rank.
    """
    # Broadcast is implemented as push + pull in BytePS
    # TODO: to make it a real broadcast, we should set the non-root tensors all 0.
    if name is None and not _executing_eagerly():
        name = 'BytePSBroadcast_%s' % _normalize_name(tensor.name)
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor(ctypes.c_char_p(name))
    return C_LIB.byteps_push_pull(tensor, name=name)


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
