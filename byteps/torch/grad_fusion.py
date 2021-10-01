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
#
# This was originally implemented by Yijie Zheng <zhengyijie@bytedance.com>,
# Yong Li <liyong.0517@bytedance.com>, and Bo Jiang <jiangbo.jacob@bytedance.com>.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import batched_fuse_, batched_unfuse_
from byteps.torch.ops import synchronize, declare
from byteps.torch.ops import size, rank
from byteps.torch.functions import broadcast_parameters as bcast_param_indices
from byteps.torch.functions import _get_loss_scale

import os
import torch

###########################################################
def recursive_traverse_grad_fn(fn, seen_fns, seen_params):
    ''' tranverse a grad fn recursively. '''
    if fn in seen_fns:
        return
    seen_fns.add(fn)

    # record tensors
    if hasattr(fn, 'variable') and isinstance(fn.variable, torch.nn.Parameter):
        seen_params.add(fn.variable)

    # recursively tranverse
    if hasattr(fn, 'next_functions'):
        for u in fn.next_functions:
            if u[0] is not None:
                recursive_traverse_grad_fn(u[0], seen_fns, seen_params)
    if hasattr(fn, 'saved_tensors'):
        for t in fn.saved_tensors:
            recursive_traverse_grad_fn(t, seen_fns, seen_params)

def find_parameters(tensors):
    ''' find paramters in the autograd graph for this tensor. '''
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    grad_fns = set()
    params = set()
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        recursive_traverse_grad_fn(tensor.grad_fn, grad_fns, params)
    return params, grad_fns

def _find_tensors(obj):
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


class Bucket:
    '''bucket for tensor fusion'''
    def __init__(self):
        self.tensor = None
        self.param_idx = list()
        self.ready_param_idx = set()
        self.name = None # for bps push_pull
        self.no_copy = False

    def init_tensor(self, num_element, dtype, no_copy=False):
        '''init bucket tensor'''
        if no_copy:
            self.no_copy = no_copy
        else:
            self.tensor = torch.zeros(num_element, dtype=dtype).cuda()

    @property
    def ready(self):
        '''check whether this bucket is ready'''
        return (len(self.param_idx) > 0 and
                len(self.ready_param_idx) == len(self.param_idx))

class _GradFusion:
    ''' implements gradient fusion during gradients reduction'''
    def __init__(self, model, optimizer):
        self._optim = optimizer
        self._init_tensor_fusion(model)

    def _init_tensor_fusion(self, model, **kwargs):
        self._fwd_handle = model.register_forward_hook(self._model_post_fwd_hook)
        self.fusion_already_reordered = False
        self._trainable_params = list(p for _, p in model.named_parameters() if p.requires_grad)
        # assert set(self._trainable_params) == set(self._parameters), \
        #        "optimzer params and model params must be the same"
        self.param_to_name = {v.__hash__(): name for name, v in
                model.named_parameters()}
        # there is an issue when using torch.Tensor as key, so use its hash instead
        # https://github.com/pytorch/pytorch/issues/7733
        self.param_idx = {v.__hash__(): i for i, v
                          in enumerate(self._trainable_params)}
        self.param_set = set(self._trainable_params)
        self.num_successful_step = 0 # tracks the number of successful training steps, 1-based index
        # tracks the order a grad hook is called within one step. Will be set to
        # 0 before each backward pass.
        self.param_order_base_idx = 0
        self.prev_step_params = set() # for recovery from error
        # the next bucket idx to allreduce
        self.curr_bucket_idx = 0
        self._init_bucket(**kwargs)
        self._init_param_reorder_state(**kwargs)

        self.bucket_set_idx = 0
        self.bucket_set = [self.buckets]

        if self._optim._staleness == 1:
            self._init_bucket(**kwargs)
            self.bucket_set.append(self.buckets)
            self.buckets = self.bucket_set[self.bucket_set_idx]

        print(f'rank {rank()}: Grad fusion enabled, num params ' +
                f'{len(self._trainable_params)}, bucket_bytes_cap {self.bucket_bytes_cap/1024/1024} MB, ' +
                f'grad_accum_step {self._optim.backward_passes_per_step} fuse_last_bucket ' +
                f'{self.fuse_last_bucket}')

        self._grad_hooks = list()

    def _construct_buckets(self):
        # construct buckets based on dtype
        # - divide parameters into buckets
        # - for each bucket record the memember params it contains
        # - for each bucket allocate a buffer to hold the fused data
        # TODO(yulu): try our best to make sure:
        #    num_bytes_in_bucket <= 1.5 * bucket_bytes_cap
        dtypes = set(v.dtype for v in self._trainable_params)
        # dtype2bucket[dtype][1] means num_element
        dtype2bucket = {v: [Bucket(), 0] for v in dtypes}
        for idx, param in enumerate(self._trainable_params):
            (bucket, num_element) = dtype2bucket[param.dtype]
            self.param_bucket_info[idx][1] = num_element
            bucket.param_idx.append(idx)
            num_element += param.numel()
            num_bytes = num_element * param.element_size()
            if num_bytes >= self.bucket_bytes_cap:
                # no_copy = len(bucket.param_idx) == 1
                no_copy = False
                bucket.init_tensor(num_element, param.dtype, no_copy = no_copy)
                self.buckets.append(bucket)
                bucket = Bucket()
                num_element = 0
            dtype2bucket[param.dtype] = [bucket, num_element]

        for dtype, (bucket, num_element) in dtype2bucket.items():
            if len(bucket.param_idx) > 0:
                bucket.init_tensor(num_element, dtype)
                self.buckets.append(bucket)

    def _init_bucket(self, **kwargs):
        '''init bucket'''
        # TODO(zhengyijie): find the best bucket_bytes_cap
        # 8*1024*1024 = 8388608
        default_bucket_size_bytes = int(os.getenv("BYTEPS_BUCKET_SIZE_BYTES",
                                                   "8388608"))
        self.bucket_bytes_cap = kwargs.get('bucket_bytes_cap',
                                           default_bucket_size_bytes)
        # the following two combined are to wait for the `fuse_last_bucket`
        # number of buckets closest to the loss to be ready , then allreduce
        # them. Note: the bucket idx 0 is the one next to the loss, i.e. first
        # one computed.
        self.fuse_bucket_num = 0
        self.fuse_last_bucket = kwargs.get('fuse_last_bucket', 0)
        # param_bucket_info[0] means bucket_idx and param_bucket_info[1] means offset
        self.param_bucket_info = [[-1, -1] for _ in self._trainable_params]
        self.buckets = list()
        self._construct_buckets()

        # Sort resulting buckets by the minimum tensor index they include.
        # We assume that the order of the tensors is the order in which they are
        # used (or the reverse order in which their gradients are produced).
        # This sorting step ensures that the buckets are ready in consecutive order.
        self.buckets = sorted(self.buckets, key=lambda k: min(k.param_idx), reverse=True)
        self.fuse_last_bucket = min(self.fuse_last_bucket, len(self.buckets))
        for i, bucket in enumerate(self.buckets):
            for param_idx in bucket.param_idx:
                self.param_bucket_info[param_idx][0] = i
            bucket.name = "Bucket.{}.step.{}".format(i, self.num_successful_step)
            declare(bucket.name) # declare tensors
        if rank() == 0:
            print("=================")
            print("bucket structure")
            for i, bucket in enumerate(self.buckets):
                print(f'bucket.name: {bucket.name}, size: {bucket.tensor.element_size() * bucket.tensor.numel()}')
                for param_idx in bucket.param_idx:
                    curr_param = self._trainable_params[param_idx]
                    print(f'  param_idx: {param_idx}, name: {self.param_to_name[curr_param.__hash__()]}, '
                          f'size: {curr_param.element_size() * curr_param.numel()}')

    def _init_param_reorder_state(self, **kwargs):
        ''' init state for reorder params. '''
        self.param_order_idx = dict()
        self.param_order_base_idx = 0
        default_probe_steps = int(os.getenv("BYTEPS_BUCKET_PROBE_STEPS", 5))
        self.param_order_record_step = kwargs.get('param_order_record_step',
            default_probe_steps)

    def _model_post_fwd_hook(self, module, _inputs, outputs):
        ''' post model forward hook. '''
        if not module.training:
            return
        self.num_successful_step += 1
        self.curr_bucket_idx = 0
        self.fuse_bucket_num = 0
        out_tensors = list(_find_tensors(outputs))

        # recover from errors.
        if len(self.prev_step_params) != 0:
            # some params in the autograd graph in the previous iter didn't
            # trigger the grad hook in the backward pass, errors happend. Fused
            # last bucket to ensure the commucation doesn't begin when error
            # happend, so handle should be empty
            assert len(self._optim._handles) == 0
            self.num_successful_step -= 1

        # reorder params to make sure the backward pass overlaps with
        # communication. It works like this:
        # 1) train for a few steps, record the actual grad finish order.
        # 2) rank 0 broadcasts the new pram idices to the world
        # 3) every worker reconstructs its buckets based on the newly received
        #    param indices.
        self.param_order_base_idx = 0
        if self.num_successful_step == self.param_order_record_step:
            self._reorder_params()
            self._init_bucket()
            self.fusion_already_reordered = True
            if self._optim._staleness:
                niter = self._optim.state['byteps_niter']
                self.bucket_set_idx = niter % (self._optim._staleness + 1)
                self.bucket_set = [self.buckets]
                self._init_bucket()
                self.bucket_set.append(self.buckets)
                self.buckets = self.bucket_set[self.bucket_set_idx]

        params_in_graph, _ = find_parameters(out_tensors)
        self.prev_step_params = params_in_graph
        skiped_params = self.param_set - params_in_graph
        for p in skiped_params:
            self._mark_param_ready(p)

    def _mark_param_ready(self, p):
        '''mark param ready'''
        if self.num_successful_step % self._optim.backward_passes_per_step == 0:
            self._push_to_bucket(p)

    def _reorder_params(self):
        ''' reorder params. '''
        param_num = len(self._trainable_params)
        sort_fn = lambda idx: self.param_order_idx.get(idx, param_num)
        param_idx = sorted(list(range(param_num)), key=sort_fn, reverse=True)
        idx_tensor = torch.tensor(param_idx)  # pylint:disable=not-callable
        name = "reorder_idx." + str(self.num_successful_step)
        bcast_param_indices([(name, idx_tensor)], root_rank=0, prefix="Reorder_params.")
        if rank() != 0:
            param_idx = idx_tensor.tolist()

        self._trainable_params = [self._trainable_params[idx] for idx in param_idx]
        self.param_idx = {v.__hash__(): i for i, v
                          in enumerate(self._trainable_params)}

    def _process_single_bucket_async(self, bucket_id):
        '''process single bucket'''
        tensor = self.buckets[bucket_id].tensor
        tensor_compressed, ctx = self._optim._compression.compress(tensor)
        niter = self._optim.state['byteps_niter']
        handle = byteps_push_pull(tensor_compressed,
                                  average=True,
                                  name=self.buckets[bucket_id].name,
                                  version=niter,
                                  staleness=self._optim._staleness)
        if self._optim._staleness == 1 and niter >= self._optim._pipesgd_warmup_iter \
                and self.fusion_already_reordered:
            self._optim._stale_handles[niter % (self._optim._staleness + 1)][bucket_id] = (handle, ctx)
        else:
            self._optim._handles[bucket_id] = (handle, ctx)

    def _batched_fuse_grad(self, bucket):
        if bucket.no_copy:
            lone_param_idx = bucket.param_idx[0]
            lone_param = self._trainable_params[lone_param_idx]
            if lone_param.grad is None:
                with torch.no_grad():
                    lone_param.grad = lone_param.new(lone_param.size()).zero_()
            bucket.tensor = lone_param.grad()
            return
        my_grads = []
        for param_idx in bucket.param_idx:
            p = self._trainable_params[param_idx]
            if p.grad is None:
                with torch.no_grad():
                    p.grad = p.new(p.size()).zero_()
            my_grads.append(p.grad)
        batched_fuse_(my_grads, bucket.tensor)

    def _switch_bucket_set(self):
        '''Current implementation only works when there are only 2 bucket sets.'''
        self.bucket_set_idx = 1 - self.bucket_set_idx
        self.buckets = self.bucket_set[self.bucket_set_idx]

    def _push_to_bucket(self, p):
        '''push param to bucket'''
        param_idx = self.param_idx[p.__hash__()]
        (bucket_id, offset) = self.param_bucket_info[param_idx]
        bucket = self.buckets[bucket_id]
        bucket.ready_param_idx.add(param_idx)
        if not bucket.ready:
            return
        self._batched_fuse_grad(bucket)
        if bucket_id < self.fuse_last_bucket:
            self.fuse_bucket_num += 1

        # fuse_last_bucket to prevent the communication has
        # started when an error occurs in the calculation of the gradient

        # wait until the first `fuse_last_bucket` number of buckets to be ready,
        # then allreduce them. `fuse_bucket_num` tracks how many buckets are
        # ready. why is this needed?
        # yulu: this assumes the missing grads are all in the first few buckets.
        if self.fuse_bucket_num < self.fuse_last_bucket:
            return
        # this bucket is full and should be processed
        while (self.curr_bucket_idx < len(self.buckets) and
               self.buckets[self.curr_bucket_idx].ready):
            self._process_single_bucket_async(self.curr_bucket_idx)
            self.curr_bucket_idx += 1

    def _clear_tensor_fusion(self, _optimizer):
        ''' clear all hook for dist parallel. '''
        if not self._optim._enable_tensor_fusion:
            return
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        for grad_hook_hanle in self._grad_hooks:
            grad_hook_hanle.remove()
        self._grad_hooks = []
        self._grad_accs = []

    def _fused_synchronize(self):
        ''' sync all. for missing grad. '''
        for bucket_idx, (handle, ctx) in self._optim._handles.items():
            compressed_output = synchronize(handle)
            output = self._optim._compression.decompress(compressed_output, ctx)
            bucket = self.buckets[bucket_idx]
            bucket.tensor = output
            my_grads = []
            for param_idx in bucket.param_idx:
                offset = self.param_bucket_info[param_idx][1]
                my_grads.append(self._trainable_params[param_idx].grad)
            if not bucket.no_copy:
                batched_unfuse_(bucket.tensor, my_grads)
            self.buckets[bucket_idx].ready_param_idx.clear()
        self._optim._handles.clear()
        self._optim.state['byteps_niter'] += 1

    def _fused_stale_synchronize(self):
        niter = self._optim.state['byteps_niter']
        self._optim._handles = self._optim._stale_handles[(niter - 1) % (self._optim._staleness + 1)]
        curr_loss_scale = _get_loss_scale()
        prev_loss_scale = self._optim.state['byteps_stale_scale'] if niter > 0 else curr_loss_scale
        grad_ratio = curr_loss_scale / prev_loss_scale
        self._switch_bucket_set() # use the other bucket set to recover stale grads

        for bucket_idx, (handle, ctx) in self._optim._handles.items():
            compressed_output = synchronize(handle)
            output = self._optim._compression.decompress(compressed_output, ctx)
            bucket = self.buckets[bucket_idx]
            bucket.tensor = output * grad_ratio
            my_grads = []
            for param_idx in bucket.param_idx:
                offset = self.param_bucket_info[param_idx][1]
                my_grads.append(self._trainable_params[param_idx].grad)
            if not bucket.no_copy:
                batched_unfuse_(bucket.tensor, my_grads)
            self.buckets[bucket_idx].ready_param_idx.clear()
        self._optim._handles.clear()
        self._optim.state['byteps_stale_scale'] = curr_loss_scale
        self._optim.state['byteps_niter'] += 1

    def _make_hook(self, p):
        def fusion_hook(*ignore):
            if self.num_successful_step < self.param_order_record_step:
                self.param_order_base_idx += 1
                idx = self.param_idx[p.__hash__()]
                org_idx = self.param_order_idx.get(idx, 0)
                cur_idx = self.param_order_base_idx
                self.param_order_idx[idx] = max(org_idx, cur_idx)

            if p in self.prev_step_params:
                self.prev_step_params.remove(p)
            self._mark_param_ready(p)
        return fusion_hook
