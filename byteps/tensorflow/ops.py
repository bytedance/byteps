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

from byteps.tensorflow.compression import Compression
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
get_pushpull_speed = _basics.get_pushpull_speed
get_telemetry = _basics.get_telemetry
session_size = int(os.environ.get('BYTEPS_ALLTOALL_SESSION_SIZE', 2))

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
            import warnings
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
    Adasum = "Adasum"

Average = ReduceOps.Average
Sum = ReduceOps.Sum
Adasum = ReduceOps.Adasum

handle_average_backwards_compatibility = get_average_backwards_compatibility_fun(ReduceOps)


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)

def randomString(stringLength=16):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def _push_pull(tensor, scope='', name=None, op=Average):
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
    return C_LIB.byteps_push_pull(tensor, name=name, input_name = full_name,
                                  op=op.value.lower())

def _alltoall(tensor, scope='', name=None, splits=None, recv_splits=None, with_size=False,
              compression=Compression.none):
    def is_group(tensor):
        if isinstance(tensor, list): return True
        if isinstance(tensor, tuple): return True
        return False              
    assert splits is not None
    # For now, `splits` is required.
    if name is None and not _executing_eagerly():
        if is_group(tensor):
            name = 'BytePSAlltoAll_%s' % _normalize_name(tensor[0].name)
        else:
            name = 'BytePSAlltoAll_%s' % _normalize_name(tensor.name)
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
    full_name = full_name.encode("ascii")

    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall.restype = None
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall.argtypes = ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int
    p = (ctypes.c_int*session_size)()
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall(full_name, p, session_size)
    tensor_key = list(p)
    if recv_splits is None:
        recv_split_unknown = True
        recv_splits = splits    
    else:
        recv_split_unknown = False

    if is_group(tensor):
        assert with_size is False, "alltoall with a list of tensors does not support with_size=True"
        # For now, the returned received splits count is not supported.
        tensors = []
        for t in tensor:
            t_compressed, dtype = compression.compress(t)
            tensors.append(t_compressed)
        # TensorFlow requires the shape of input tensors to be identical.
        # You should guarantee this before calling this op.
        recved_data, recved_size = C_LIB.byteps_alltoall_group(tensors, splits=splits, recv_splits=recv_splits,
                                                  name=name, input_name=full_name,
                                                  recv_split_unknown=recv_split_unknown, tensor_key=tensor_key)
        tensors_decompressed = []
        for i in range(len(recved_data)):
            tensors_decompressed.append(compression.decompress(recved_data[i], dtype))
        return tensors_decompressed
    else: # single tensor
        # compress if needed
        tensor_compressed, dtype = compression.compress(tensor)
        recved_data, recved_size = C_LIB.byteps_alltoall(tensor_compressed, splits=splits, recv_splits=recv_splits,
                                                        name=name, input_name=full_name,
                                                        recv_split_unknown=recv_split_unknown, tensor_key=tensor_key)
        tensor_decompressed = compression.decompress(recved_data, dtype)
    if not with_size:
        return tensor_decompressed
    # TODO: recved_size returned from the op might not be set when recv_split is given. Here we directly
    # return recv_splits back to the user instead
    if not recv_split_unknown:
        recved_size = recv_splits
    return tensor_decompressed, recved_size

def _alltoall_cpu2gpu(tensor, scope='', name=None, splits=None, recv_splits=None,
                      with_size=False, compression=Compression.none):
    def is_group(tensor):
        if isinstance(tensor, list): return True
        if isinstance(tensor, tuple): return True
        return False
    
    assert splits is not None
    # For now, `splits` is required.
    if name is None and not _executing_eagerly():
        if is_group(tensor):
            name = 'BytePSAlltoAll_%s' % _normalize_name(tensor[0].name)
        else:
            name = 'BytePSAlltoAll_%s' % _normalize_name(tensor.name)
    if scope == '' and not _executing_eagerly():
        if 'v1' in dir(tf.compat):
            scope = tf.compat.v1.get_default_graph().get_name_scope()
        else:
            scope = tf.get_default_graph().get_name_scope()
        if scope != '':
            scope += '/'
    if not name:
        name = ''
    full_name = scope + name + "_cpu2gpu"
    full_name = full_name.encode("ascii")

    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall.restype = None
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall.argtypes = ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int
    p = (ctypes.c_int*session_size)()
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall(full_name, p, session_size)
    tensor_key = list(p)
    if recv_splits is None:
        recv_split_unknown = True
        recv_splits = splits    
    else:
        recv_split_unknown = False

    if is_group(tensor):
        assert with_size is False, "alltoall with a list of tensors does not support with_size=True"
        # For now, the returned received splits count is not supported.
        tensors = []
        for t in tensor:
            # compress if needed
            t_compressed, dtype = compression.compress(t)
            tensors.append(t_compressed)
        # TensorFlow requires the shape of input tensors to be identical.
        # You should guarantee this before calling this op.
        recved_data, recved_size = C_LIB.byteps_alltoall_cputogpu_group(tensors, splits=splits, recv_splits=recv_splits,
                                                  name=name, input_name=full_name,
                                                  recv_split_unknown=recv_split_unknown, tensor_key=tensor_key)
        tensors_decompressed = []
        for i in range(len(recved_data)):
            tensors_decompressed.append(compression.decompress(recved_data[i], dtype))
        return tensors_decompressed
    else: # single tensor
        # compress if needed
        tensor_compressed, dtype = compression.compress(tensor)
        recved_data, recved_size = C_LIB.byteps_alltoall_cputogpu(tensor_compressed, splits=splits, recv_splits=recv_splits,
                                                        name=name, input_name=full_name,
                                                        recv_split_unknown=recv_split_unknown, tensor_key=tensor_key)
        tensor_decompressed = compression.decompress(recved_data, dtype)

    if not with_size:
        return tensor_decompressed
    # TODO: recved_size returned from the op might not be set when recv_split is given. Here we directly
    # return recv_splits back to the user instead
    if not recv_split_unknown:
        recved_size = recv_splits
    return tensor_decompressed, recved_size


def _alltoall_gpu2cpu(tensor, scope='', name=None, splits=None, recv_splits=None,
                      with_size=False, compression=Compression.none):
    def is_group(tensor):
        if isinstance(tensor, list): return True
        if isinstance(tensor, tuple): return True
        return False
    assert splits is not None
    # For now, `splits` is required.
    if name is None and not _executing_eagerly():
        name = 'BytePSAlltoAll_%s' % _normalize_name(tensor.name)
    if scope == '' and not _executing_eagerly():
        if 'v1' in dir(tf.compat):
            scope = tf.compat.v1.get_default_graph().get_name_scope()
        else:
            scope = tf.get_default_graph().get_name_scope()
        if scope != '':
            scope += '/'
    if not name:
        name = ''
    full_name = scope + name + "_gpu2cpu"
    full_name = full_name.encode("ascii")

    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall.restype = None
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall.argtypes = ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int
    p = (ctypes.c_int*session_size)()
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_alltoall(full_name, p, session_size)
    tensor_key = list(p)
    if recv_splits is None:
        recv_split_unknown = True
        recv_splits = splits    
    else:
        recv_split_unknown = False

    if is_group(tensor):
        assert with_size is False, "alltoall with a list of tensors does not support with_size=True"
        # For now, the returned received splits count is not supported.
        tensors = []
        for t in tensor:
            # compress if needed
            t_compressed, dtype = compression.compress(t)
            tensors.append(t_compressed)
        # TensorFlow requires the shape of input tensors to be identical.
        # You should guarantee this before calling this op.
        recved_data, recved_size = C_LIB.byteps_alltoall_gputocpu_group(tensors, splits=splits, recv_splits=recv_splits,
                                                  name=name, input_name=full_name,
                                                  recv_split_unknown=recv_split_unknown, tensor_key=tensor_key)
        tensors_decompressed = []
        for i in range(len(recved_data)):
            tensors_decompressed.append(compression.decompress(recved_data[i], dtype))
        return tensors_decompressed
    else: # single tensor
        # compress if needed
        tensor_compressed, dtype = compression.compress(tensor)
        recved_data, recved_size = C_LIB.byteps_alltoall_gputocpu(tensor_compressed, splits=splits, recv_splits=recv_splits,
                                                        name=name, input_name=full_name,
                                                        recv_split_unknown=recv_split_unknown, tensor_key=tensor_key)
        tensor_decompressed = compression.decompress(recved_data, dtype)
    
    if not with_size:
        return tensor_decompressed
    # TODO: recved_size returned from the op might not be set when recv_split is given. Here we directly
    # return recv_splits back to the user instead
    if not recv_split_unknown:
        recved_size = recv_splits
    return tensor_decompressed, recved_size


@ops.RegisterGradient('BytepsAlltoall')
def _alltoall_grad(op, grad, recv_bytes):
    """Gradient for alltoall op.
    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.
    Returns:
      The gradient with respect to the input of the op.
    """
    tensor = op.inputs[0]
    splits = op.inputs[1]
    recv_splits = op.inputs[2]
    name = op.get_attr('input_name').decode() + '_bwd_'
    # FIXME: this might not work if recv_splits is not provided
    result = _alltoall(grad, splits=recv_splits, recv_splits=splits, name=name)
    return [result, None, None]

@ops.RegisterGradient('BytepsAlltoallGroup')
def _alltoall_group_grad(op, *outputs):
    """Gradients for alltoall_group op.
    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.
    Returns:
      The gradients with respect to the input of the op.
    """
    n = len(outputs)
    grads = outputs[0:int(n/2)]
    splits = op.inputs[-2]
    recv_splits = op.inputs[-1]
    name = op.get_attr('input_name').decode() + '_bwd_'
    # FIXME: this might not work if recv_splits is not provided
    results = _alltoall(grads, splits=recv_splits, recv_splits=splits, name=name)
    return list(results) + [None, None] 

@ops.RegisterGradient('BytepsAlltoallCputogpu')
def _alltoall_cpu2gpu_grad(op, grad, recv_bytes):
    """Gradient for alltoall op.
    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.
    Returns:
      The gradient with respect to the input of the op.
    """
    tensor = op.inputs[0]
    splits = op.inputs[1]
    recv_splits = op.inputs[2]
    name = op.get_attr('input_name').decode() + '_bwd_'
    # FIXME: this might not work if recv_splits is not provided
    result = _alltoall_gpu2cpu(grad, splits=recv_splits, recv_splits=splits, name=name)
    return [result, None, None]

@ops.RegisterGradient('BytepsAlltoallCputogpuGroup')
def _alltoall_cpu2gpu_group_grad(op, *outputs):
    """Gradients for alltoall_group op.
    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.
    Returns:
      The gradients with respect to the input of the op.
    """
    n = len(outputs)
    grads = outputs[0:int(n/2)]
    splits = op.inputs[-2]
    recv_splits = op.inputs[-1]
    name = op.get_attr('input_name').decode() + '_bwd_'
    # FIXME: this might not work if recv_splits is not provided
    results = _alltoall_gpu2cpu(grads, splits=recv_splits, recv_splits=splits, name=name)
    return list(results) + [None, None] 


@ops.RegisterGradient('BytepsAlltoallGputocpu')
def _alltoall_gpu2cpu_grad(op, grad, recv_bytes):
    """Gradient for alltoall op.
    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.
    Returns:
      The gradient with respect to the input of the op.
    """
    tensor = op.inputs[0]
    splits = op.inputs[1]
    recv_splits = op.inputs[2]
    name = op.get_attr('input_name').decode() + '_bwd_'
    # FIXME: this might not work if recv_splits is not provided
    result = _alltoall_cpu2gpu(grad, splits=recv_splits, recv_splits=splits, name=name)
    return [result, None, None]

@ops.RegisterGradient('BytepsAlltoallGputocpuGroup')
def _alltoall_gpu2cpu_group_grad(op, *outputs):
    """Gradients for alltoall_group op.
    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.
    Returns:
      The gradients with respect to the input of the op.
    """
    n = len(outputs)
    grads = outputs[0:int(n/2)]
    splits = op.inputs[-2]
    recv_splits = op.inputs[-1]
    name = op.get_attr('input_name').decode() + '_bwd_'
    # FIXME: this might not work if recv_splits is not provided
    results = _alltoall_cpu2gpu(grads, splits=recv_splits, recv_splits=splits, name=name)
    return list(results) + [None, None] 

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


def _do_recv_async(tensor, sender, name, scope='', version=0, priority=0):
    receiver = -1
    if name is None and not _executing_eagerly():
        name = 'BytePSRecv_%s' % _normalize_name(tensor.name)
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
    full_name = full_name.encode()
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_p2p(full_name, sender, receiver)
    C_LIB.byteps_recv(tensor, sender=sender, receiver=receiver, input_name=full_name,
                      version=version, priority=priority)

def _do_send_async(tensor, receiver, name, scope='', version=0, priority=0):
    sender = -1
    if name is None and not _executing_eagerly():
        name = 'BytePSSend_%s' % _normalize_name(tensor.name)
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
    full_name = full_name.encode()
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor_p2p(full_name, sender, receiver)
    C_LIB.byteps_send(tensor, sender=sender, receiver=receiver, input_name=full_name,
                      version=version, priority=priority)

def recv_async(tensor, sender, name=None, version=0, priority=0):
    """An op which receives tensor from sender rank to the input tensor inplace.
    Arguments:
        tensor: A tensor to receive tensor.
        sender: A number to indicate which process is the sender.
        name: A name bound to input tensor.
        version: normally not used.
        prority: normally not used.
    """
    return _do_recv_async(tensor, sender, name,
                          version=version, priority=priority)

def send_async(tensor, receiver, name=None, version=0, priority=0):
    """An op which sends input tensor to receiver process.
    Arguments:
        tensor: A tensor to be sent.
        receiver: A number to indicate which process is the receiver.
        name: A name bound to input tensor.
        version: normally not used.
        prority: normally not used.
    """
    return _do_send_async(tensor, receiver, name,
                          version=version, priority=priority)
