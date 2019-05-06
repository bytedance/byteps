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

# PyTorch v2 API starts with 1.0.0 (including nightly builds)
_v2_api = LooseVersion(torch.__version__) >= LooseVersion('1.0.0')
if _v2_api:
    from byteps.common import BytePSBasics as _BytePSBasics
    from byteps.torch import c_lib
    _basics = _BytePSBasics(__file__, 'c_lib')
    _NULL = ""
else:
    # TODO: we may not support older pytorch. Raise exception here
    pass

from byteps.torch.compression import Compression

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank


# Schema: handle -> input, output
# We keep input in order to make sure it does not get garbage collected
# before the operation is finished.
_handle_map = {}

# Only support fp16 allreduce for PyTorch versions using v2 API.
_fp16_supported = _v2_api


def _check_function(function_factory, tensor):
    function = function_factory(tensor)
    if not hasattr(c_lib, function):
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if not tensor.is_contiguous():
        raise ValueError('Tensor is required to be contiguous.')
    return function


def _push_pull_function_factory(tensor):
    return 'byteps_torch_push_pull_async_' + tensor.type().replace('.', '_')


def _push_pull_async(tensor, output, average, name):
    if tensor.dtype == torch.float16 and not _fp16_supported:
        raise NotImplementedError(
            'float16 allreduce is not supported for PyTorch version {} < 1.0.0'
            .format(torch.__version__))

    function = _check_function(_push_pull_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                        name.encode() if name is not None else _NULL)
    _handle_map[handle] = (tensor, output)
    return handle


def push_pull_async(tensor, average=True, name=None):
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
        A handle to the allreduce operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _push_pull_async(tensor, output, average, name)


class BytePSAllreduce(torch.autograd.Function):
    """An autograd function that performs allreduce on a tensor."""

    @staticmethod
    def forward(ctx, tensor, average, name):
        ctx.average = average
        handle = push_pull_async(tensor, average, name)
        return synchronize(handle)

    @staticmethod
    def backward(ctx, grad_output):
        return push_pull(grad_output, ctx.average), None, None


def push_pull(tensor, average=True, name=None, compression=Compression.none):
    """
    A function that performs averaging or summation of the input tensor over all the
    BytePS processes. The input tensor is not modified.
    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
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
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    tensor_compressed, ctx = compression.compress(tensor)
    summed_tensor_compressed = BytePSAllreduce.apply(tensor_compressed, average, name)
    return compression.decompress(summed_tensor_compressed, ctx)


def push_pull_async_(tensor, average=True, name=None):
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
        A handle to the allreduce operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _push_pull_async(tensor, tensor, average, name)


def push_pull_(tensor, average=True, name=None):
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
    handle = push_pull_async_(tensor, average, name)
    return synchronize(handle)


def _broadcast_function_factory(tensor):
    return 'byteps_torch_broadcast_async_' + tensor.type().replace('.', '_')


def _broadcast_async(tensor, output, root_rank, name):
    function = _check_function(_broadcast_function_factory, tensor)
    handle = getattr(c_lib, function)(
        tensor, output, root_rank, name.encode() if name is not None else _NULL)
    _handle_map[handle] = (tensor, output)
    return handle


def broadcast_async(tensor, root_rank, name=None):
    """
    A function that asynchronously broadcasts the input tensor on root rank to the same
    input tensor on all other BytePS processes. The input tensor is not modified.
    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.
    Returns:
        A handle to the broadcast operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _broadcast_async(tensor, output, root_rank, name)


class BytePSBroadcast(torch.autograd.Function):
    """An autograd function that broadcasts a tensor."""

    @staticmethod
    def forward(ctx, tensor, root_rank, name):
        ctx.root_rank = root_rank
        handle = broadcast_async(tensor, root_rank, name)
        return synchronize(handle)

    @staticmethod
    def backward(ctx, grad_output):
        grad_reduced = push_pull(grad_output, average=False)
        if rank() != ctx.root_rank:
            grad_reduced *= 0
        return grad_reduced, None, None


def broadcast(tensor, root_rank, name=None):
    """
    A function that broadcasts the input tensor on root rank to the same input tensor
    on all other BytePS processes. The input tensor is not modified.
    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.
    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.
    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.
    Returns:
        A tensor of the same shape and type as `tensor`, with the value broadcasted
        from root rank.
    """
    return BytePSBroadcast.apply(tensor, root_rank, name)


def broadcast_async_(tensor, root_rank, name=None):
    """
    A function that asynchronously broadcasts the input tensor on root rank to the same
    input tensor on all other BytePS processes. The operation is performed in-place.
    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.
    Returns:
        A handle to the broadcast operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _broadcast_async(tensor, tensor, root_rank, name)


def broadcast_(tensor, root_rank, name=None):
    """
    A function that broadcasts the input tensor on root rank to the same input tensor
    on all other BytePS processes. The operation is performed in-place.
    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.
    Returns:
        A tensor of the same shape and type as `tensor`, with the value broadcasted
        from root rank.
    """
    handle = broadcast_async_(tensor, root_rank, name)
    return synchronize(handle)


def poll(handle):
    """
    Polls an allreduce, allgather or broadcast handle to determine whether underlying
    asynchronous operation has completed. After `poll()` returns `True`, `synchronize()`
    will return without blocking.
    Arguments:
        handle: A handle returned by an allreduce, allgather or broadcast asynchronous
                operation.
    Returns:
        A flag indicating whether the operation has completed.
    """
    return c_lib.byteps_torch_poll(handle) != 0


def synchronize(handle):
    """
    Synchronizes an asynchronous allreduce, allgather or broadcast operation until
    it's completed. Returns the result of the operation.
    Arguments:
        handle: A handle returned by an allreduce, allgather or broadcast asynchronous
                operation.
    Returns:
        An output tensor of the operation.
    """
    if handle not in _handle_map:
        return
    c_lib.byteps_torch_wait_and_clear(handle)
    _, output = _handle_map.pop(handle)
    return output