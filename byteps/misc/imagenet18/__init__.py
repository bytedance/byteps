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

        self._parameter_map = {}
        for fp16_p, fp32_p in zip(self.fp16_params, self.fp32_params):
            if self._is_tensor_instance:
                self._parameter_map[fp32_p.__hash__()] = fp16_p
            else:
                self._parameter_map[fp32_p] = fp16_p

        self.backward_passes_per_step = backward_passes_per_step
        self._push_pull_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()

        # Use lock to block the forward propagation of each parameter.
        self._locks = {}
        for param_group in self.param_groups:
            for p in param_group['params']:
                self._locks[p] = threading.Lock()

        if size() > 1:
            self._register_forward_hooks()
            self._register_hooks()

        # declare tensors
        for name in self._parameter_names.values():
            declare("Gradient."+name)
        # We use two loops for load-balancing
        for name in self._parameter_names.values():
            declare("Parameter."+name)

        # Poll whether the tensor push-pull is finished.
        self._event_queue = queue.Queue()
        self._poller = threading.Thread(target=self._poll, args=())
        self._poller.start()

    def __del__(self):
        """Clean up"""
        self._event_queue.put((None, None, None))
        self._poller.join()

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def zero_grad(self):
        """Override the default zero_grad function.
        Clears the gradients of all optimized tensors.
        """
        if size() > 1 and self._step > 0:
            return
        else:
            super(self.__class__, self).zero_grad()

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._push_pull_delay:
            self._push_pull_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    if self._is_tensor_instance:
                        fp16_p = self._parameter_map.get(p.__hash__())
                    else:
                        fp16_p = self._parameter_map.get(p)
                    p_tmp = fp16_p.expand_as(fp16_p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _push_pull_grad_async(self, p):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
            fp16_p = self._parameter_map.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
            fp16_p = self._parameter_map.get(p)
        tensor = fp16_p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)
        handle = byteps_push_pull(tensor_compressed, average=True, name="Gradient."+name)
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

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = self._push_pull_grad_async(p)
            self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx = self._push_pull_grad_async(p)
                self._handles[p] = (handle, ctx)
        
        while len(self._handles.keys()) > 0:
            params = list(self._handles.keys())
            for p in params:
                handle, _ = self._handles[p]
                if poll(handle):
                    output = synchronize(handle)
                    self._push_pull_delay[p] = self.backward_passes_per_step
                    if self._is_tensor_instance:
                        fp16_p = self._parameter_map.get(p.__hash__())
                    else:
                        fp16_p = self._parameter_map.get(p)
                    fp16_p.grad.set_(self._compression.decompress(output, ctx))
                    if p.grad is None:
                        p.grad = Variable(p.data.new(*p.data.size()))
                    p.grad.data.copy_(fp16_p.grad.data)
                    p.grad.data = p.grad.data / self.loss_scale
                    self.step_one_param(p)
                    self._handles.pop(p)

        self._handles.clear()

    def step(self, closure=None):
        self.synchronize()
        return None

    def step_one_param(self, param, closure=None):
        """Performs a single optimization step.
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

    def _poll(self):
        """Poll the completion of the tensor's backward or push-pull from a FIFO event_queue"""
        while True:
            p, handle, ctx = self._event_queue.get()
            if p is None:
                break
            # Check whether the push-pull is finished. If so, start updating parameters.
            if handle is not None and poll(handle):
                output = synchronize(handle)
                p.grad.set_(self._compression.decompress(output, ctx))
                self._push_pull_delay[p] = self.backward_passes_per_step
                self.step_one_param(p)
                self._zero_one_grad(p)
                # notify update completion and parameter is ready for forward propagation
                if p in self._locks:
                    self._locks[p].release()
            else:
                self._event_queue.put((p, handle, ctx))

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
                if p in self._handles:
                    del self._handles[p]
                if p not in self._locks:
                    continue
                with self._locks[p]:
                    pass

        # Register pre-hook and hook for each module
        for mod in reversed(submodules):
            mod.register_forward_pre_hook(pre_forward_hook)

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as torch accumulates gradients by default.
        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

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

