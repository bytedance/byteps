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
import os
# PyTorch must be >= 1.0.0 (including nightly builds)
# This should be guaranteed by setup.py
# TODO: we may not support older pytorch. Raise exception here
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

P2P_VERBOSE = bool(os.environ.get('BYTEPS_P2P_VERBOSE', False))

# Schema: handle -> input, output
# We keep input in order to make sure it does not get garbage collected
# before the operation is finished.
_handle_map = {}


def _check_function(function_factory, tensor):
    function = function_factory(tensor)
    from byteps.torch import c_lib
    if not hasattr(c_lib, function):
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if tensor.is_sparse:
        raise ValueError('Tensor is required to be dense.')
    return function


def _push_pull_function_factory(tensor):
    return 'byteps_torch_push_pull_async_' + tensor.type().replace('.', '_')

def _send_function_factory(tensor):
    return 'byteps_torch_send_async_' + tensor.type().replace('.', '_')

def _recv_function_factory(tensor):
    return 'byteps_torch_recv_async_' + tensor.type().replace('.', '_')

def _push_pull_group_function_factory(tensor):
    return 'byteps_torch_push_pull_group_sync_' + tensor.type().replace('.', '_')

def _batched_fuse_function_factory(tensor):
    return 'byteps_torch_batched_fuse_async_' + tensor.type().replace('.', '_')

def _batched_unfuse_function_factory(tensor):
    return 'byteps_torch_batched_unfuse_async_' + tensor.type().replace('.', '_')

def _batched_zero_out_function_factory(tensor):
    return 'byteps_torch_batched_zero_out_async_' + tensor.type().replace('.', '_')

def _delay_compensation_function_factory(tensor):
    return 'byteps_torch_delay_compensation_async_' + tensor.type().replace('.', '_')

def _dc_adam_function_factory(tensor):
    return 'byteps_torch_dc_adam_async_' + tensor.type().replace('.', '_')

def _allgather_function_factory(tensor):
    return 'byteps_torch_allgather_async_' + tensor.type().replace('.', '_')

def _do_push_pull_async(tensor, output, average, name, version=0, priority=0, staleness=0):
    from byteps.torch import c_lib
    if staleness != 0:
        version = version % (staleness + 1)
    c_lib.byteps_torch_declare_tensor(name.encode() if name is not None else _NULL, staleness)
    function = _check_function(_push_pull_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority, staleness)
    _handle_map[handle] = (tensor, output)
    return handle

def _do_recv_async(tensor, sender, name, version=0, priority=0):
    from byteps.torch import c_lib
    receiver = -1
    c_name = name.encode() if name is not None else _NULL
    _declare_p2p(name, sender, receiver)
    
    function = _check_function(_recv_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, sender, receiver,
                                      c_name, version, priority)
    if P2P_VERBOSE:
        print(f'BPS recv {name}. {sender}->{rank()}. {tensor.size()} {tensor.dtype} handle={handle}', flush=True)
    _handle_map[handle] = (tensor, tensor)
    return handle

def _do_send_async(tensor, receiver, name, version=0, priority=0):
    from byteps.torch import c_lib
    sender = -1
    c_name = name.encode() if name is not None else _NULL
    _declare_p2p(name, sender, receiver)

    function = _check_function(_send_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, sender, receiver,
                                      c_name, version, priority)
    if P2P_VERBOSE:
        print(f'BPS send {name}. {rank()}->{receiver}. {tensor.size()} {tensor.dtype} handle={handle}', flush=True)
    _handle_map[handle] = (tensor, tensor)
    return handle

def _do_push_pull_group_sync(tensor, output, average, name, version=0, priority=0, staleness=0):
    from byteps.torch import c_lib
    if staleness != 0:
        version = version % (staleness + 1)
    c_lib.byteps_torch_declare_tensor(name.encode() if name is not None else _NULL, staleness)
    function = _check_function(_push_pull_group_function_factory, tensor)
    handle, curr_count = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority, staleness)
    _handle_map[handle] = (tensor, output)
    return handle, curr_count

def _do_allgather_async(tensor, output, shape_list, name, version=0, priority=0, staleness=0):
    from byteps.torch import c_lib
    assert staleness == 0, 'allgather not support staleness > 0'
    c_lib.byteps_torch_declare_tensor_allgather(name.encode() if name is not None else _NULL, staleness)
    function = _check_function(_allgather_function_factory, tensor)
    handle = getattr(c_lib, function)(tensor, output, shape_list,
                                      name.encode() if name is not None else _NULL,
                                      version, priority, staleness)
    _handle_map[handle] = (tensor, output)
    return handle

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
    return _do_push_pull_async(tensor, output, average, name, version, priority, staleness=0)

def recv_async(tensor, sender, name=None, version=0, priority=0):
    """
    TODO: doc
    Arguments:
        tensor: A tensor to average and sum.
        name: A name of the reduction operation.
    Returns:
        A handle to the send operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _do_recv_async(tensor, sender, name, version, priority)

def send_async(tensor, receiver, name=None, version=0, priority=0):
    """
    TODO: doc
    Arguments:
        tensor: A tensor to average and sum.
        name: A name of the reduction operation.
    Returns:
        A handle to the send operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _do_send_async(tensor, receiver, name, version, priority)

def allgather_async(tensor, shape_list=None, name=None, version=0, priority=0):
    """
    A function that asynchronously concatenates the input tensor over all 
    the BytePS processes. The input tensor is not modified. If name is not 
    provided, an incremented auto-generated name is used.  The tensor type 
    must be the same on all BytePS processes for a given name. The concatenation 
    is done on the first dimension, so the input tensors on the different processes 
    must have the same shape, except for the first dimension,  which is allowed 
    to be different. The allgather will not start until all processes are ready 
    to send and receive the tensor.
    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.
    Returns:
        A handle to the allgather operation that can be used with `poll()` or
        `synchronize()`.
    """
    shape = list(tensor.shape)
    if not shape_list:
        shape[0] *= size()
    else:
        shape[0] = sum(shape_list)

    output = tensor.new(torch.Size(shape))
    return _do_allgather_async(tensor, output, shape_list, name, version, priority, staleness=0)

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
                         ctx.average, ctx.name, ctx.version, ctx.priority), None, None, None, None


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


def push_pull_async_inplace(tensor, average=True, name=None, version=0,
                            priority=0, staleness=0):
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
        staleness: the maximum staleness for pipe SGD.
                   See DistributedOptimizer.staleness for details.
    Returns:
        A handle to the push_pull operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _do_push_pull_async(tensor, tensor, average, name, version, priority, staleness)

def push_pull_group_sync_inplace(tensor, average=True, name=None, version=0, priority=0, staleness=0):
    return _do_push_pull_group_sync(tensor, tensor, average, name, version, priority, staleness)

def push_pull_inplace(tensor, average=True, name=None, version=0, priority=0, staleness=0):
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
        staleness: the maximum staleness for pipe SGD.
                   See DistributedOptimizer.staleness for details.
    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    handle = push_pull_async_inplace(tensor, average, name, version, priority, staleness)
    return synchronize(handle)


class BytePSAllgather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, shape_list, name, version, priority):
        ctx.dim = tensor.shape[0]
        ctx.shape_list = shape_list
        ctx.name = name
        ctx.version = version
        ctx.priority = priority
        handle = allgather_async(tensor, shape_list, name, version, priority)
        return synchronize(handle)

    @staticmethod
    def backward(ctx, grad_output):
        grad_reduced = push_pull(grad_output, True, ctx.name + '_allgather_grad', ctx.version, ctx.priority)

        offset = 0
        r = rank()
        if not ctx.shape_list:
            offset = r * ctx.dim
        else:
            offset = sum(ctx.shape_list[0:r])

        return grad_reduced.narrow(0, offset, ctx.dim), None, None, None, None


def allgather(tensor, same_shape=True, name=None, version=0, priority=0):
    """
    A function that asynchronously concatenates the input tensor over all 
    the BytePS processes. The input tensor is not modified. The name must be provided.  
    The tensor type must be the same on all BytePS processes for a given name. The 
    concatenation is done on the first dimension, so the input tensors on the different 
    processes must have the same shape, except for the first dimension, which is allowed 
    to be different. The allgather will not start until all processes are ready to send 
    and receive the tensor.
    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.
    Arguments:
        tensor: A tensor to allgather.
        same_shape: Whether the tensor is with the same shape over all ranks or not.
        name: A name of the allgather operation.
        compression: Compression algorithm used during allgather to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    Returns:
        A tensor of the same type as `tensor`, concatenated on dimension zero
        across all processes. The shape is identical to the input shape, except for
        the first dimension, which may be greater and is the sum of all first
        dimensions of the tensors in different processes.
    """
    if name == None:
        raise AssertionError("To manually call allgather, you must specify a name by name=...")
    
    if len(tensor.shape) == 0:
        tensor = torch.unsqueeze(tensor, 0)

    shape_list = []
    if same_shape == False:
        name += "_V"

        d = torch.tensor([tensor.shape[0]], device=torch.device('cuda'))
        shape_list = BytePSAllgather.apply(d, shape_list, name + "_shape_list", version, priority).tolist()

        is_equal = not shape_list or shape_list.count(shape_list[0]) == len(shape_list)
        if is_equal:
            shape_list = []

    return BytePSAllgather.apply(tensor, shape_list, name, version, priority)

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
    from byteps.torch import c_lib
    return c_lib.byteps_torch_poll(handle) != 0

def batched_fuse_(src_tensors, dst_tensor):
    """
    Use batched cuda kernels to copy entries from dst_tensors to src_tensor. The
    dtype of src_tensors and dst_tensor must be the same. The input and output
    must be on the same device. The total number of elements in src_tensors must
    be less or equal than the number of elements of dst_tensor.

    Arguments:
        src_tensors: A list of tensors. Use batched cuda operation for GPU
                     tensors.
        dst_tensor: A tensor. On exit the entries from src_tensors
                    will be copied to dst_tensor.
    Returns:
        0
    """
    src_tensors = [item for item in src_tensors if item is not None]
    if not src_tensors:
        return 0

    from byteps.torch import c_lib
    if dst_tensor.is_cuda:
        function = _check_function(_batched_fuse_function_factory, dst_tensor)
        getattr(c_lib, function)(src_tensors, dst_tensor)
        return 0
    else:
        idx = 0
        dst_tensor_flat = dst_tensor.view(-1)
        for item in src_tensors:
            tmp = dst_tensor_flat[idx:idx + item.numel()]
            tmp.copy_(item.view(-1))
            idx += item.numel()
        return 0

def batched_unfuse_(src_tensor, dst_tensors):
    """
    Use batched cuda kernels to copy entries from src_tensor to dst_tensors. The
    dtype of src_tensor and dst_tensors must be the same. The input and output
    must be on the same device. The number of elements in src_tensor must be
    less or equal than the total number of elements of dst_tensors.

    Arguments:
        src_tensor: A tensor. Use batched cuda operation for GPU tensors.
        dst_tensors: A list of tensors. On exit the entries from src_tensor
                     will be copied to dst_tensors.
    Returns:
        0
    """
    dst_tensors = [item for item in dst_tensors if item is not None]
    if not dst_tensors:
        return 0

    from byteps.torch import c_lib
    if src_tensor.is_cuda:
        function = _check_function(_batched_unfuse_function_factory, src_tensor)
        getattr(c_lib, function)(src_tensor, dst_tensors)
        return 0
    else:
        src_tensor_flat = src_tensor.view(-1)
        idx = 0
        for item in dst_tensors:
            tmp = src_tensor_flat[idx:idx + item.numel()].view(item.size())
            item.copy_(tmp)
            idx += item.numel()
        return 0

def batched_zero_(tensors):
    """
    Use batched cuda kernels to set entries of GPU tensors to 0.
    Arguments:
        tensors: A list of tensors. Use batched cuda calls only for GPU tensors.
    Returns:
        0
    """
    tensors = [item for item in tensors if item is not None]
    if not tensors:
        return 0

    from byteps.torch import c_lib
    assert isinstance(tensors, list), type(tensors)
    if tensors[0].is_cuda:
        function = _check_function(_batched_zero_out_function_factory, tensors[0])
        getattr(c_lib, function)(tensors)
        return 0
    else:
        for item in tensors:
            item.zero_()
        return 0

@torch.no_grad()
def delay_compensation_(params, grads, prev_params, dc_lambda):
    """
    Update grads according to delay compensation using diagonal approximation.
    See reference https://arxiv.org/abs/1609.08326 for details.
    Arguments:
        params: A list of tensors, each of which is a parameter.
        grads: A list of tensors, each of which is the grad associated with parameters.
        prev_params: A list of tensors, each of which is params from previous iteration.
        dc_lambda: A scalar.
    Returns:
        0
    """
    from byteps.torch import c_lib
    assert isinstance(params, list), type(tensors)
    assert isinstance(grads, list), type(tensors)
    assert isinstance(prev_params, list), type(tensors)
    if params[0].is_cuda:
        function = _check_function(_delay_compensation_function_factory, params[0])
        getattr(c_lib, function)(params, grads, prev_params, dc_lambda)
        return 0
    else:
        for p, g, prev_p in zip(params, grads, prev_params):
            g += dc_lambda * g * g * (p - prev_p)
        
        for prev_p, p in zip(prev_params, params):
            prev_p.copy_(p)
        return 0

@torch.no_grad()
def dc_adam_(params, grads, prev_params, dc_lambda, exp_avgs, exp_avg_sqs, steps, lr, eps, weight_decay, beta1, beta2):
    """
    Update grads according to delay compensation using diagonal approximation.
    Arguments:
        params: A list of tensors, each of which is a parameter.
        grads: A list of tensors, each of which is the grad associated with parameters.
        prev_params: A list of tensors, each of which is params from previous iteration.
        dc_lambda: A scalar.
    Returns:
        0
    """
    from byteps.torch import c_lib
    assert isinstance(params, list), type(tensors)
    assert isinstance(grads, list), type(tensors)
    assert isinstance(prev_params, list), type(tensors)
    if params[0].is_cuda:
        function = _check_function(_dc_adam_function_factory, params[0])
        getattr(c_lib, function)(params, grads, prev_params, dc_lambda, exp_avgs, exp_avg_sqs, steps, lr, eps, weight_decay, beta1, beta2)
        return 0
    else:
        # TODO add python implementation of DC-ADAM
        raise NotImplementedError
        # for p, g, prev_p in zip(params, grads, prev_params):
        #     g += dc_lambda * g * g * (p - prev_p)
        
        # for prev_p, p in zip(prev_params, params):
        #     prev_p.copy_(p)
        return 0


def declare(name, staleness=0):
    from byteps.torch import c_lib
    c_lib.byteps_torch_declare_tensor(name.encode(), staleness)
    return 0

def _declare_p2p(name, sender, receiver):
    from byteps.torch import c_lib
    c_name = name.encode() if name is not None else _NULL
    c_lib.byteps_torch_declare_tensor_p2p(c_name, sender, receiver)
    return 0

def byteps_torch_set_num_grads(num_grads_):
    from byteps.torch import c_lib
    c_lib.byteps_torch_set_num_grads(num_grads_)
    return 0

def synchronize(handle, busy_wait=False):
    """
    Synchronizes an asynchronous push_pull operation until
    it's completed. Returns the result of the operation.
    Arguments:
        handle: A handle returned by an push_pull asynchronous
                operation.
    Returns:
        An output tensor of the operation.
    """
    from byteps.torch import c_lib
    if P2P_VERBOSE:
        print(f'synchronize handle={handle}', flush=True)
    if handle not in _handle_map:
        if P2P_VERBOSE:
            print(f'DONE synchronize handle={handle}', flush=True)
        return
    c_lib.byteps_torch_wait_and_clear(handle, busy_wait)
    _, output = _handle_map.pop(handle)
    if P2P_VERBOSE:
        print(f'DONE synchronize handle={handle}', flush=True)
    return output
