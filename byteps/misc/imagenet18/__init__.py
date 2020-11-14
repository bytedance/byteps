# Copyright 2019 Bytedance Inc. All Rights Reserved.
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

from byteps.torch import _DistributedOptimizer
from byteps.torch.compression import Compression
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import push_pull
from byteps.torch.ops import poll, synchronize, declare
from byteps.torch.ops import init, shutdown
from byteps.torch.ops import size, local_size, rank, local_rank

import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue
import torch
import collections


class _HalfPrecisionDistributedOptimizer(torch.optim.Optimizer):
    def  __init__(self, params, named_parameters, model, fp16_params, fp32_params, loss_scale,
                 compression, backward_passes_per_step=1):
        super(self.__class__, self).__init__(params)
        self._compression = compression

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._model = model
        self.fp32_params = fp32_params
        self.fp16_params = fp16_params
        self.loss_scale = loss_scale

        # Track training steps
        # self._step = 0

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = _HalfPrecisionDistributedOptimizer.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        if len(named_parameters) > 0:
            if isinstance(named_parameters[0][1], torch.Tensor):
                if any([not isinstance(p, torch.Tensor) for name, p in named_parameters]):
                    raise ValueError('named_parameters should consistently be a sequence of '
                                     'tuples (name, torch.Tensor)')
                self._is_tensor_instance = True
                # there is an issue when using torch.Tensor as key, so use its hash instead
                # https://github.com/pytorch/pytorch/issues/7733
                self._parameter_names = {v.__hash__(): k for k, v
                                         in sorted(named_parameters)}
                self._tensor_list = [tensor for name, tensor in named_parameters]
            else:
                self._is_tensor_instance = False
                self._parameter_names = {v: k for k, v
                                         in sorted(named_parameters)}
        else:
            self._is_tensor_instance = False
            self._parameter_names = {v: 'push_pull.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}

        self._fp32_to_fp16_map = {}
        self._fp16_to_fp32_map = {}
        for fp16_p, fp32_p in zip(self.fp16_params, self.fp32_params):
            if self._is_tensor_instance:
                self._fp32_to_fp16_map[fp32_p.__hash__()] = fp16_p
            else:
                self._fp32_to_fp16_map[fp32_p] = fp16_p
            self._fp16_to_fp32_map[fp16_p] = fp32_p

        self.backward_passes_per_step = backward_passes_per_step
        self._push_pull_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()

        if size() > 1:
            self._register_forward_hooks()
            self._register_backward_hooks()

        self._lock = threading.Lock()

        # declare tensors
        for name in self._parameter_names.values():
            declare("Gradient."+name)
        # We use two loops for load-balancing
        for name in self._parameter_names.values():
            declare("Parameter."+name)

        self.priorities = {}
        self.gradient_count = 0

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._push_pull_delay:
            self._push_pull_delay[p] = self.backward_passes_per_step

    def _register_forward_hooks(self):
        """Add hook before forward propagation of each layer to block forward computation until the push-pull and
        parameter update is finished. The blocking is implemented using a lock."""
        # Recursively find all submodules
        submodules = []
        q = queue.LifoQueue()
        for mod in self._model.children():
            q.put(mod)
        while not q.empty():
            mod = q.get()
            if len(list(mod.children())) == 0:
                submodules.append(mod)
            else:
                for m in mod.children():
                    q.put(m)

        def pre_forward_hook(mod, input):
            for p in mod.parameters():
                fp32_p = self._fp16_to_fp32_map[p]
                if fp32_p in self._handles:
                    while True:
                        with self._lock:
                            if fp32_p not in self._handles:
                                break
                            if self._try_to_synchronize(fp32_p):
                                break
                            else:
                                # In order not to waste GPU cycles, we find another handle to synchronize
                                params = list(self._handles.keys())
                                for other_p in params:
                                    if other_p.__hash__() == fp32_p.__hash__():
                                        continue
                                    if self._try_to_synchronize(other_p):
                                        break
                        time.sleep(0.0001)

        def after_forward_hook(mod, input, result):
            for p in mod.parameters():
                self._zero_one_grad(p)

        # Register pre-hook and hook for each module
        for mod in reversed(submodules):
            mod.register_forward_pre_hook(pre_forward_hook)
            mod.register_forward_hook(after_forward_hook)

    def _register_backward_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    if self._is_tensor_instance:
                        fp16_p = self._fp32_to_fp16_map.get(p.__hash__())
                    else:
                        fp16_p = self._fp32_to_fp16_map.get(p)
                    p_tmp = fp16_p.expand_as(fp16_p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _push_pull_grad_async(self, p):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
            fp16_p = self._fp32_to_fp16_map.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
            fp16_p = self._fp32_to_fp16_map.get(p)
        tensor = fp16_p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)
        if fp16_p not in self.priorities:
            self.priorities[fp16_p] = self.gradient_count
            self.gradient_count += 1
        handle = byteps_push_pull(tensor_compressed, average=False, name="Gradient."+name, priority=self.priorities[fp16_p])
        return handle, ctx

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._push_pull_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._push_pull_delay[p] > 0
            handle, ctx = None, None
            self._push_pull_delay[p] -= 1
            if self._push_pull_delay[p] == 0:
                handle, ctx = self._push_pull_grad_async(p)
            self._handles[p] = (handle, ctx)
        return hook

    def _sync_missing_gradients(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = self._push_pull_grad_async(p)
            self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx = self._push_pull_grad_async(p)
                self._handles[p] = (handle, ctx)

    def step(self, closure=None, wait_for_finish=True):
        if size() > 1:
            self._sync_missing_gradients()
            if wait_for_finish:
                self._wait_for_all()
            loss = None
            if closure is not None:
                loss = closure()
            return loss
        else:
            super(self.__class__, self).step()


    def _step_one_param(self, param, closure=None):
        """Performs a single optimization step. Only applicable to SGD.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.__hash__() != param.__hash__():
                    continue
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as torch accumulates gradients by default.
        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    def _wait_for_all(self):
        while len(self._handles.keys()) > 0:
            params = list(self._handles.keys())
            for p in params:
                self._try_to_synchronize(p)

    def _try_to_synchronize(self, p):
        handle, ctx = self._handles[p]
        if poll(handle):
            output = synchronize(handle)
            self._push_pull_delay[p] = self.backward_passes_per_step
            if self._is_tensor_instance:
                fp16_p = self._fp32_to_fp16_map.get(p.__hash__())
            else:
                fp16_p = self._fp32_to_fp16_map.get(p)
            fp16_p.grad.set_(self._compression.decompress(output, ctx))
            p.grad.data.copy_(fp16_p.grad.data)
            p.grad.data = p.grad.data / (self.loss_scale * size())
            self._step_one_param(p)
            fp16_p.data.copy_(p.data)
            self._handles.pop(p)
            return True
        else:
            return False


def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1,
                         half=False,
                         model=None,
                         fp16_params=None,
                         fp32_params=None,
                         loss_scale=1024):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an push_pull to
    average gradient values before applying gradients to model weights.
    push_pull operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all push_pull operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces push_pull operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.
    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          push_pull operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during push_pull to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an push_pull implementation.
    if half:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                   dict(_HalfPrecisionDistributedOptimizer.__dict__))
        return cls(optimizer.param_groups, named_parameters, model,
                   fp16_params, fp32_params, loss_scale, compression, backward_passes_per_step)
    else:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                   dict(_DistributedOptimizer.__dict__))
        return cls(optimizer.param_groups, named_parameters,
                   compression, backward_passes_per_step)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.
    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run synchronous broadcasts.
    for name, p in params:
        # Broadcast is implemented as push + pull in BytePS
        # To make it a real broadcast, we set the non-root tensors all 0.
        if rank() != root_rank:
            p.fill_(0)
        # Remember to disable averaging because we are doing broadcast
        handle = byteps_push_pull(p, average=False, name="Parameter."+name)
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.
    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces push_pull on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(
                option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
