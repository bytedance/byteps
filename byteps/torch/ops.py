# Copyright 2019 ByteDance, Inc. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from distutils.version import LooseVersion

# Load all the necessary PyTorch C types.
import torch

# PyTorch must be >= 1.0.0 (including nightly builds)
# This should be guaranteed by setup.py
# TODO: we may not support older pytorch. Raise exception here
from byteps.torch import c_lib
from byteps.common import BytePSBasics as _BytePSBasics
_basics = _BytePSBasics(__file__, 'c_lib')
_NULL = ""


from byteps.torch.compression import Compression

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
suspend = _basics.suspend
resume = _basics.resume
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank


# Schema: handle -> input, output
# We keep input in order to make sure it does not get garbage collected
# before the operation is finished.
_handle_map = {}


def _check_function(function_factory, tensor):
    function = function_factory(tensor)
    if not hasattr(c_lib, function):
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if not tensor.is_contiguous():
        raise ValueError('Tensor is required to be contiguous.')
    return function


def _push_pull_function_factory(tensor):
    return 'byteps_torch_push_pull_async_' + tensor.type().replace('.', '_')

def _push_pull_group_function_factory(tensor):
    return 'byteps_torch_push_pull_group_sync_' + tensor.type().replace('.', '_')

def _do_push_pull_async(tensor, output, average, name, version=0, priority=0):
    c_lib.byteps_torch_declare_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_push_pull_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority)
    _handle_map[handle] = (tensor, output)
    return handle

def _do_push_pull_group_sync(tensor, output, average, name, version=0, priority=0):
    c_lib.byteps_torch_declare_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_push_pull_group_function_factory, tensor)
    handle, curr_count = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority)
    _handle_map[handle] = (tensor, output)
    return handle, curr_count


def push_pull_async(tensor, average=True, name=None, version=0, priority=0):
    """
    A function that performs asynchronous averaging or summation of the input tensor
    over all the BytePS processes. The input tensor is not modified.
    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
    Returns:
        A handle to the push_pull operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _do_push_pull_async(tensor, output, average, name, version, priority)


class BytePSPushPull(torch.autograd.Function):
    """An autograd function that performs push_pull on a tensor."""

    @staticmethod
    def forward(ctx, tensor, average, name, version, priority):
        ctx.average = average
        ctx.name = name
        ctx.version = version
        ctx.priority = priority
        handle = push_pull_async(tensor, average, name, version, priority)
        return synchronize(handle)

    @staticmethod
    def backward(ctx, grad_output):
        return push_pull(grad_output,
                         ctx.average, ctx.name, ctx.version, ctx.priority), None, None


def push_pull(tensor, average=True, name=None, version=0, priority=0, compression=Compression.none):
    """
    A function that performs averaging or summation of the input tensor over all the
    BytePS processes. The input tensor is not modified. The reduction operation is keyed
    by the name. The name must be provided. The tensor type and shape must be the same on all
    BytePS processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.
    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.
    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
        compression: Compression algorithm used during push_pull to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    if name == None:
        raise AssertionError("To manually call push_pull, you must specify a name by name=...")
    tensor_compressed, ctx = compression.compress(tensor)
    summed_tensor_compressed = BytePSPushPull.apply(
        tensor_compressed, average, name, version, priority)
    return compression.decompress(summed_tensor_compressed, ctx)


def push_pull_async_inplace(tensor, average=True, name=None, version=0, priority=0):
    """
    A function that performs asynchronous in-place averaging or summation of the input
    tensor over all the BytePS processes.
    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
    Returns:
        A handle to the push_pull operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _do_push_pull_async(tensor, tensor, average, name, version, priority)

def push_pull_group_sync_inplace(tensor, average=True, name=None, version=0, priority=0):
    return _do_push_pull_group_sync(tensor, tensor, average, name, version, priority)

def push_pull_inplace(tensor, average=True, name=None, version=0, priority=0):
    """
    A function that performs in-place averaging or summation of the input tensor over
    all the BytePS processes.
    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    handle = push_pull_async_inplace(tensor, average, name, version, priority)
    return synchronize(handle)


"""
intra_gather is the a function to perform the gather operation among nodes in the same machine before decompression.
The task flow is as follows:
GPU compress -> gather
"""
# -------zhuang: intra_gather start-------
def _intra_gather_function_factory(tensor):
    return 'byteps_torch_intra_gather_async_' + tensor.type().replace('.', '_')


def _do_intra_gather_async(tensor, output, average=False, name=None, version=0, priority=0, root=0):
    c_lib.byteps_torch_declare_intra_gather_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_intra_gather_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority, root)
    _handle_map[handle] = (tensor, output)
    return handle


def intra_gather_async(tensor, average=False, name=None, version=0, priority=0, root=0):
    output_shape = list(tensor.shape)
    # assume all workers have the same data size
    output_shape[0] *= local_size()
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    return _do_intra_gather_async(tensor, output, average, name, version, priority, root)


def intra_gather(tensor, average=False, name=None, version=0, priority=0, root=0):
    handle = intra_gather_async(tensor, average, name, version, priority, root)
    return synchronize(handle)
# -------zhuang: intra_gather end-------

"""
intra_broadcast is the a function to perform the broadcast operation among nodes in the same machine
"""
# -------zhuang: intra_broadcast start-------
def _intra_broadcast_function_factory(tensor):
    return 'byteps_torch_intra_broadcast_async_' + tensor.type().replace('.', '_')


def _do_intra_broadcast_async(tensor, output, average=False, name=None, version=0, priority=0, root=0):
    c_lib.byteps_torch_declare_intra_broadcast_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_intra_broadcast_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority, root)
    _handle_map[handle] = (tensor, output)
    return handle


def intra_broadcast_async(tensor, average=False, name=None, version=0, priority=0, root=0):
    return _do_intra_broadcast_async(tensor, tensor, average, name, version, priority, root)

# for root node, tensor is the send buff; for non-root nodes, tensor is the recv buff
def intra_broadcast(tensor, average=False, name=None, version=0, priority=0, root=0):
    handle = intra_broadcast_async(tensor, average, name, version, priority, root)
    return synchronize(handle)
# -------zhuang: intra_broadcast end-------


"""
intra_reduce is the a function to perform the reduce operation among nodes in the same machine.
"""
# -------zhuang: intra_reduce start-------
def _intra_reduce_function_factory(tensor):
    return 'byteps_torch_intra_reduce_async_' + tensor.type().replace('.', '_')


def _do_intra_reduce_async(tensor, output, average=True, name=None, version=0, priority=0, root=0):
    c_lib.byteps_torch_declare_intra_reduce_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_intra_reduce_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority, root)
    _handle_map[handle] = (tensor, output)
    return handle

# intra_reduce is an inplace operation
def intra_reduce_async(tensor, average=True, name=None, version=0, priority=0, root=0):
    return _do_intra_reduce_async(tensor, tensor, average, name, version, priority, root)


def intra_reduce(tensor, average=True, name=None, version=0, priority=0, root=0):
    handle = intra_reduce_async(tensor, average, name, version, priority, root)
    return synchronize(handle)
# -------zhuang: intra_reduce end-------

"""
intra_reducescatter is the a function to perform the reducescatter operation among nodes in the same machine.
"""
# -------zhuang: intra_reducescatter start-------
def _intra_reducescatter_function_factory(tensor):
    return 'byteps_torch_intra_reducescatter_async_' + tensor.type().replace('.', '_')


def _do_intra_reducescatter_async(tensor, output, average=True, name=None, version=0, priority=0):
    c_lib.byteps_torch_declare_intra_reducescatter_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_intra_reducescatter_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority)
    _handle_map[handle] = (tensor, output)
    return handle


def intra_reducescatter_async(tensor, average=True, name=None, version=0, priority=0):
    return _do_intra_reducescatter_async(tensor, tensor, average, name, version, priority)

# note that the return of intra_reducescatter is the original tensor 
def intra_reducescatter(tensor, average=True, name=None, version=0, priority=0):
    handle = intra_reducescatter_async(tensor, average, name, version, priority)
    return synchronize(handle)
# -------zhuang: intra_reduce end-------

"""
intra_allgather is the a function to perform the allgather operation among nodes in the same machine.
"""
# -------zhuang: intra_allgather start-------
def _intra_allgather_function_factory(tensor):
    return 'byteps_torch_intra_allgather_async_' + tensor.type().replace('.', '_')


def _do_intra_allgather_async(tensor, output, average=False, name=None, version=0, priority=0):
    c_lib.byteps_torch_declare_intra_allgather_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_intra_allgather_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority)
    _handle_map[handle] = (tensor, output)
    return handle

def intra_allgather_async(tensor, average=False, name=None, version=0, priority=0):
    output_shape = list(tensor.shape)
    # assume all workers have the same data size
    output_shape[0] *= local_size()
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    return _do_intra_allgather_async(tensor, output, average, name, version, priority)


def intra_allgather(tensor, average=False, name=None, version=0, priority=0):
    handle = intra_allgather_async(tensor, average, name, version, priority)
    return synchronize(handle)
# -------zhuang: intra_allgather end-------

"""
intra_alltoall is the a function to perform the alltoall operation among nodes in the same machine.
"""
# -------zhuang: intra_alltoall start-------
def _intra_alltoall_function_factory(tensor):
    return 'byteps_torch_intra_alltoall_async_' + tensor.type().replace('.', '_')


def _do_intra_alltoall_async(tensor, output, average=False, name=None, version=0, priority=0):
    c_lib.byteps_torch_declare_intra_alltoall_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_intra_alltoall_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority)
    _handle_map[handle] = (tensor, output)
    return handle


# in-place alltoall. This API requires all involved GPUs have the same message size
def intra_alltoall_async(tensor, average=False, name=None, version=0, priority=0):
    output = torch.empty_like(tensor)
    return _do_intra_alltoall_async(tensor, output, average, name, version, priority)


def intra_alltoall(tensor, average=False, name=None, version=0, priority=0):
    handle = intra_alltoall_async(tensor, average, name, version, priority)
    return synchronize(handle)
# -------zhuang: intra_alltoall end-------


"""
cpu_compress is the a function to perform the reduce operation for compressed messages across machines.
The task flow is as follows:
intra-node communication -> CPU compress -> PUSH -> PULL -> CPU decompress -> intra-node communication broadcast
"""
# -------zhuang: cpu_compress start-------
def _cpu_compress_function_factory(tensor):
    return 'byteps_torch_cpu_compress_async_' + tensor.type().replace('.', '_')


def _do_cpu_compress_async(tensor, output, average=False, name=None, version=0, priority=0):
    c_lib.byteps_torch_declare_cpu_compress_tensor(name.encode() if name is not None else _NULL, tensor.numel())
    function = _check_function(_cpu_compress_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority)
    _handle_map[handle] = (tensor, output)
    return handle


def cpu_compress_async(tensor, average=False, name=None, version=0, priority=0):
    return _do_cpu_compress_async(tensor, tensor, average, name, version, priority)


def cpu_compress(tensor, average=False, name=None, version=0, priority=0):
    handle = cpu_compress_async(tensor, average, name, version, priority)
    return synchronize(handle)
# -------zhuang: cpu_compresse end-------



def poll(handle):
    """
    Polls an push_pull handle to determine whether underlying
    asynchronous operation has completed. After `poll()` returns `True`, `synchronize()`
    will return without blocking.
    Arguments:
        handle: A handle returned by an push_pull asynchronous
                operation.
    Returns:
        A flag indicating whether the operation has completed.
    """
    return c_lib.byteps_torch_poll(handle) != 0


def declare(name):
    c_lib.byteps_torch_declare_tensor(name.encode())
    return 0

def byteps_torch_set_num_grads(num_grads_):
    c_lib.byteps_torch_set_num_grads(num_grads_)
    return 0

def synchronize(handle):
    """
    Synchronizes an asynchronous push_pull operation until
    it's completed. Returns the result of the operation.
    Arguments:
        handle: A handle returned by an push_pull asynchronous
                operation.
    Returns:
        An output tensor of the operation.
    """
    if handle not in _handle_map:
        return
    c_lib.byteps_torch_wait_and_clear(handle)
    _, output = _handle_map.pop(handle)
    return output
