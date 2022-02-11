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

import itertools
from contextlib import contextmanager

from byteps.torch.compression import Compression
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import batched_zero_
from byteps.torch.ops import poll, synchronize, declare
from byteps.torch.ops import size, rank
from byteps.torch.grad_fusion import _GradFusion
from byteps.torch.functions import _get_loss_scale

import os
import torch
import collections

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression,
                 backward_passes_per_step=1, staleness=0,
                 pipesgd_warmup_iter=0,
                 model=None, **kwargs):
        super(self.__class__, self).__init__(params)
        self._compression = compression
        self._staleness = staleness
        self._pipesgd_warmup_iter = pipesgd_warmup_iter
        assert staleness == 0 or staleness == 1, staleness

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._enable_async = (int(os.getenv('BYTEPS_ENABLE_ASYNC', 0)) != 0)
        if self._enable_async:
            assert int(os.getenv('DMLC_NUM_WORKER')) > 1, \
                "Async is only valid for distributed training"
            print('BytePS: enable asynchronous training')

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = _DistributedOptimizer.find_duplicates([k for k, _ in named_parameters])
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
                self._parameters = self._tensor_list
            else:
                self._is_tensor_instance = False
                self._parameter_names = {v: k for k, v
                                         in sorted(named_parameters)}
                self._parameters = [v for k, v in sorted(named_parameters)]
        else:
            self._is_tensor_instance = False
            self._parameter_names = {v: 'push_pull.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
            self._parameters = [v for param_group in self.param_groups
                                  for i, v in enumerate(param_group['params'])]
        self.backward_passes_per_step = backward_passes_per_step
        self._push_pull_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._should_sync = True

        # for pipesgd
        self._stale_handles = collections.defaultdict(dict)
        self._stale_handles[-1] = {}
        self._stale_push_pull_delay = collections.defaultdict(dict)
        self._stale_push_pull_delay = {self._get_param_name(v): self.backward_passes_per_step
                                       for _, v in sorted(named_parameters)}
        # whether it is the initial optimizer.step()
        self.state['byteps_skipped_init_step'] = False
        # a reference of self._niter stored in states
        self.state['byteps_niter'] = 0
        # store the scale factor from amp if fp16 dynamic scaling is present
        self.state['byteps_stale_scale'] = 1
        # checkpoint the materialized gradient before torch.save() is called.
        # the ckpt grad will be used in three cases: the iteration after checkpoint,
        # the 1-st iteration after resuming training from ckpt, and the first iteration
        # that pipesgd takes effect.
        self.state['byteps_stale_grad'] = {}

        # for tensor fusion
        self._model = model
        self._enable_tensor_fusion = False
        if self._model and size() > 1:
            self._enable_tensor_fusion = True
            self._grad_fusion = _GradFusion(model, self)

        if size() > 1:
            self._register_hooks()

        # declare tensors
        for name in sorted(self._parameter_names.values()):
            declare("Gradient."+name, staleness=staleness)
        # We use two loops for load-balancing
        for name in sorted(self._parameter_names.values()):
            declare("Parameter."+name)

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
        for p in self._stale_push_pull_delay:
            self._stale_push_pull_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _get_param_name(self, p):
        """Get the name of a parameter."""
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        return name

    def _push_pull_grad_async(self, p):
        name = self._get_param_name(p)
        if self._enable_async:
            # the real handle will be created in step()
            handle, ctx = None, None
        else:
            tensor = p.grad
            tensor_compressed, ctx = self._compression.compress(tensor)
            # pipe-sgd: need to clone the gradient to avoid race condition
            niter = self.state['byteps_niter']
            if self._staleness and niter >= self._pipesgd_warmup_iter:
                tensor_compressed = tensor_compressed.clone()
            handle = byteps_push_pull(tensor_compressed, average=True, name="Gradient."+name,
                                      version=niter, staleness=self._staleness)
        return handle, ctx, name

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
            handle, ctx, name = None, None, None
            self._push_pull_delay[p] -= 1
            if self._push_pull_delay[p] == 0:
                handle, ctx, name = self._push_pull_grad_async(p)
            self._handles[p] = (handle, ctx, name)

        def stale_hook(*ignore):
            name = self._get_param_name(p)
            niter = self.state['byteps_niter']
            if name in self._stale_handles[niter] and self._stale_handles[niter][name][0] is not None:
                if self._stale_push_pull_delay[name] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            handle, ctx = None, None
            self._stale_push_pull_delay[name] -= 1
            if self._stale_push_pull_delay[name] == 0:
                handle, ctx, name = self._push_pull_grad_async(p)
                self._stale_handles[niter][name] = (p, handle, ctx)

        if self._enable_tensor_fusion and size() > 1:
            return self._grad_fusion._make_hook(p)
        elif self._staleness == 0:
            return hook
        else:
            return stale_hook

    def synchronize(self):
        """Synchronizes the asynchronous pushpull(allreduce) operations for all
        gradients until they are completed.
        """
        if self._enable_tensor_fusion:
            niter = self.state['byteps_niter']
            if self._staleness and niter >= self._pipesgd_warmup_iter \
                    and self._grad_fusion.fusion_already_reordered:
                self._grad_fusion._fused_stale_synchronize()
            else:
                self._grad_fusion._fused_synchronize()
        elif self._staleness:
            self._stale_synchronize()
        else:
            self._synchronize()

    def _synchronize(self):
        """Synchronize the pushpull operations"""
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            if type(p.grad) == type(None):
                continue
            handle, ctx, name = self._push_pull_grad_async(p)
            self._handles[p] = (handle, ctx, name)

        for p, value in self._handles.items():
            handle, ctx, name = value
            if handle is None:
                handle, ctx, _ = self._push_pull_grad_async(p)
                self._handles[p] = (handle, ctx, name)
        for p, (handle, ctx, name) in self._handles.items():
            output = synchronize(handle)
            self._push_pull_delay[p] = self.backward_passes_per_step
            if not self._enable_async:
                tmp = self._compression.decompress(output, ctx)
                if self._compression == Compression.none:
                    p.grad.set_(tmp)
                else:
                    p.grad.copy_(tmp)
        self._handles.clear()

    def _stale_synchronize(self):
        """Synchronize the pushpull operations when pipesgd is enabled"""
        has_amp = hasattr(self, "_amp_stash")
        niter = self.state['byteps_niter']
        assert niter >= 0, niter
        loss_scale = _get_loss_scale()
        # if the loss scale increases at the current iteration
        # amp will rescale it back after synchronize(). so we need
        # to adjust the gradient from the previous step accordingly
        if niter > self._pipesgd_warmup_iter:
            prev_loss_scale = self.state['byteps_stale_scale']
        else:
            prev_loss_scale = loss_scale
        grad_ratio = loss_scale / prev_loss_scale

        # materialzed grad tensors are not available. obtain them from handles
        stale_grad_state = self.state['byteps_stale_grad']
        if not stale_grad_state:
            if niter <= self._pipesgd_warmup_iter:
                if niter == self._pipesgd_warmup_iter:
                    print(f'BytePS pipeSGD: started pipeline at iter {niter}', flush=True)
                for name, (p, handle, ctx) in self._stale_handles[niter].items():
                    assert handle is not None, name
                    assert not p.grad.is_sparse, "sparse gradient is not supported"
                    output = synchronize(handle)
                    tmp = self._compression.decompress(output, ctx)
                    # sync SGD duration warmup
                    if niter < self._pipesgd_warmup_iter:
                        p.grad.data = tmp
                    else:
                        stale_grad_state[name] = tmp
                        p.grad.copy_(tmp)
            else:
                for name, (p, handle, ctx) in self._stale_handles[niter-1].items():
                    assert handle is not None
                    assert not p.grad.is_sparse, "sparse gradient is not supported"
                    output = synchronize(handle)
                    prev_grad = self._compression.decompress(output, ctx)
                    with torch.no_grad():
                        # if the loss scale increases at the current iteration
                        # amp will rescale it back after synchronize(). so we need
                        # to adjust the gradient from the previous step accordingly
                        if grad_ratio != 1.0:
                            prev_grad.mul_(grad_ratio)
                        p.grad.copy_(prev_grad)
            if (niter - 1) in self._stale_handles:
                del self._stale_handles[niter - 1]
        else:
            # grad tensors alread materialized
            for p in self._requires_update:
                assert not p.grad.is_sparse, "sparse gradient is not supported"
                name = self._get_param_name(p)
                if name in stale_grad_state:
                    prev_grad = stale_grad_state[name]
                    with torch.no_grad():
                        if grad_ratio != 1.0:
                            prev_grad.mul_(grad_ratio)
                        p.grad.copy_(prev_grad)
            self.state['byteps_stale_grad'] = {}

        # update states
        for name in self._stale_push_pull_delay:
            self._stale_push_pull_delay[name] = self.backward_passes_per_step
        self.state['byteps_stale_scale'] = loss_scale
        self.state['byteps_niter'] += 1

    def prepare_stale_states(self):
        """
        This API is used to save _stale_grad and _stale_scale when both checkpointing
        and PipeSGD are enabled. The ckpt _stale_grad and _stale_scale will be used for
        update when resuming training from ckpt. Please Note: User must call this API intentionally
        before torch.save().
        """
        stale_grad_states = {}
        niter = self.state['byteps_niter']
        for name, (p, handle, ctx) in self._stale_handles[niter-1].items():
            assert handle is not None
            assert not p.grad.is_sparse, p
            output = synchronize(handle)
            prev_grad = self._compression.decompress(output, ctx)
            stale_grad_states[name] = prev_grad
        self.state['byteps_stale_grad'] = stale_grad_states
        del self._stale_handles[niter-1]
        for name in self._stale_push_pull_delay:
            self._stale_push_pull_delay[name] = self.backward_passes_per_step

    @contextmanager
    def skip_synchronize(self):
        if self._enable_async:
            raise AssertionError("skip_synchronize cannot be used in async training")
        self._should_sync = False
        try:
            yield
        finally:
            self._should_sync = True

    def step(self, closure=None):
        if self._enable_async:
            old_weight_map = {}
            # store the weights before update
            for p, _ in self._handles.items():
                old_weight_map[p] = p.data.clone().detach()
            # update
            loss = super(self.__class__, self).step(closure)

            for p, (h, _) in self._handles.items():
                # get the diff for each weight (in-place)
                p.data.sub_(old_weight_map.get(p))
                if h is None:
                    # create the handler now
                    if self._is_tensor_instance:
                        name = self._parameter_names.get(p.__hash__())
                    else:
                        name = self._parameter_names.get(p)
                    handle = byteps_push_pull(p, average=False, name="AsyncParam."+name)
                    _, ctx = self._compression.compress(p)
                    self._handles[p] = (handle, ctx)

            self.synchronize()
            return loss
        else:
            # skip sync if calling skip_synchronize
            if self._should_sync:
                self.synchronize()
            niter = self.state['byteps_niter']
            # synchronize() already incremented niter by 1
            pipesgd_active = self._staleness and niter > self._pipesgd_warmup_iter

            if self._enable_tensor_fusion:
                if not pipesgd_active or not self._grad_fusion.fusion_already_reordered:
                    return super(self.__class__, self).step(closure)
                else:
                    if not self.state['byteps_skipped_init_step']:
                        self.state['byteps_skipped_init_step'] = True
                    else:
                        return super(self.__class__, self).step(closure)
            elif pipesgd_active:
                if not self.state['byteps_skipped_init_step']:
                    self.state['byteps_skipped_init_step'] = True
                else:
                    return super(self.__class__, self).step(closure)
            else:
                return super(self.__class__, self).step(closure)

    def zero_grad(self):
        """
        Use batched GPU kernel launch to zero out all grads using only one
        kernel launch.
        """
        my_grads = []
        for p in self._parameters:
            if type(p.grad) == type(None):
                continue
            my_grads.append(p.grad)
        batched_zero_(my_grads)

def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1, staleness=0,
                         pipesgd_warmup_iter=0,
                         model=None, **kwargs):
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
        staleness: Number of controlled gradients staleness if pipelined SGD is enabled. 
                   This allows optimizer using stale gradients to update parameters. Defaults 
                   to not using pipelined SGD, i.e., staleness=0. If set to 1, the parameter
                   update is delayed by 1 step. Reference: https://arxiv.org/abs/1811.03619
        pipesgd_warmup_iter: Number of warmup steps for pipesgd, during which pipesgd staleness
                   is fixed at 0.
        model: The model being trained. Passing the model in enables tensor
               fusion.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an push_pull implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               compression, backward_passes_per_step,
               staleness, pipesgd_warmup_iter=pipesgd_warmup_iter,
               model=model, **kwargs)
