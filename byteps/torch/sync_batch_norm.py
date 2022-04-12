# Based on https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/_functions.py
# Modifications copyright 2020 Maka Autonomous Robotic Systems
# Copyright 2021 Bytedance Inc. All Rights Reserved.
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

from byteps.torch.ops import allgather_async, push_pull_async, size, rank, synchronize

from distutils.version import LooseVersion

import torch
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


# Backward compat for old PyTorch
if not hasattr(torch.jit, 'unused'):
    torch.jit.unused = lambda x: x


_SYNC_BN_V2 = (
    LooseVersion(torch.__version__) >= LooseVersion('1.5.0') and
    LooseVersion(torch.__version__) <= LooseVersion('1.6.0')
)
_SYNC_BN_V3 = LooseVersion(torch.__version__) >= LooseVersion('1.6.0')
_SYNC_BN_V4 = LooseVersion(torch.__version__) >= LooseVersion('1.9.0')


class SyncBatchNorm(_BatchNorm):
    """Applies synchronous version of N-dimensional BatchNorm.

    In this version, normalization parameters are synchronized across workers during forward pass.
    This is very useful in situations where each GPU can fit a very small number of examples.

    See https://pytorch.org/docs/stable/nn.html#batchnorm2d for more details about BatchNorm.

    Arguments:
        num_features: number of channels `C` from the shape `(N, C, ...)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to `None` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to `True`, this module has
            learnable affine parameters. Default: `True`
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: `True`

    .. note:: Only GPU input tensors are supported in the training mode.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
            track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'.format(input.dim()))

    def _run_bn(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

    @torch.jit.unused
    def _maybe_run_sync_bn(self, input):
        if size() == 1:
            return self._run_bn(input)
        return _SyncBatchNorm.apply(
            input, self.weight, self.bias, self.running_mean, self.running_var,
            self.eps, self.momentum)

    def forward(self, input):
        # currently only GPU input is supported by underlying kernel from PyTorch
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1

        if not self.training and self.track_running_stats:
            return self._run_bn(input)
        else:
            return self._maybe_run_sync_bn(input)

    @classmethod
    def convert_sync_batchnorm(cls, module):
        '''
        Recursively traverse module and its children to replace all instances of
        ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`bps.parallel.SyncBatchNorm`.
        All ``torch.nn.BatchNorm*N*d`` wrap around
        ``torch.nn.modules.batchnorm._BatchNorm``, so this function lets you easily switch
        to use sync BN.
        Args:
            module (torch.nn.Module): input module
        Example::
            >>> # model is an instance of torch.nn.Module
            >>> import byteps.torch as bps
            >>> sync_bn_model = bps.SyncBatchNorm.convert_sync_batchnorm(model)
        '''
        mod = module
        if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
            return module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            mod = cls(module.num_features,
                                module.eps,
                                module.momentum,
                                module.affine,
                                module.track_running_stats)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            mod.num_batches_tracked = module.num_batches_tracked
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()
        for name, child in module.named_children():
            mod.add_module(name, cls.convert_sync_batchnorm(child))
        # TODO(jie) should I delete model explicitly?
        del module
        return mod

class _SyncBatchNorm(Function):
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum):
        input = input.contiguous()

        my_size = input.numel() // input.size(1)
        count = torch.tensor([my_size]).cuda()

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        shape_list = []
        name_count = 'sync_batch_norm.count.' + str(torch.numel(count)) + '.' + str(count.dtype)
        name_mean = 'sync_batch_norm.mean.' + str(torch.numel(mean)) + '.' + str(mean.dtype)
        name_invstd = 'sync_batch_norm.invstd.' + str(torch.numel(invstd)) + '.' + str(invstd.dtype)
        count_handle = allgather_async(count, shape_list=shape_list, name=name_count)
        mean_handle = allgather_async(mean.unsqueeze(0), shape_list=shape_list, name=name_mean)
        invstd_handle = allgather_async(invstd.unsqueeze(0), shape_list=shape_list, name=name_invstd)
        # wait on the async communication to finish
        count_all = synchronize(count_handle)
        mean_all = synchronize(mean_handle)
        invstd_all = synchronize(invstd_handle)

        if _SYNC_BN_V3:
            counts_for_bngswc = count_all.view(-1).float().to(input.device)
        else:
            # backwards compatibility
            counts_for_bngswc = count_all.view(-1).tolist()

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            counts_for_bngswc
        )

        self.save_for_backward(input, weight, mean, invstd, count_all)

        # apply element-wise normalization
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_all = self.saved_tensors
        need_input_grad, need_weight_grad, need_bias_grad = self.needs_input_grad[0:3]

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            need_input_grad,
            need_weight_grad,
            need_bias_grad
        )

        if need_input_grad:
            name_sum_dy = 'sync_batch_norm.sum_dy.' + str(torch.numel(sum_dy)) + '.' + str(sum_dy.dtype)
            name_sum_dy_xmu = 'sync_batch_norm.sum_dy_xmu.' + str(torch.numel(sum_dy_xmu)) + '.' + str(sum_dy_xmu.dtype)
            sum_dy_handle = push_pull_async(sum_dy, average=False, name=name_sum_dy)
            sum_dy_xmu_handle = push_pull_async(sum_dy_xmu, average=False, name=name_sum_dy_xmu)

            # wait on the async communication to finish
            sum_dy = synchronize(sum_dy_handle)
            sum_dy_xmu = synchronize(sum_dy_xmu_handle)

            if _SYNC_BN_V4:
                # from 1.9.0 on we need a count tensor on all devices
                # count_all is calculated as total count across all ranks in forward function
                count_all = count_all.to(dtype=torch.int, device=grad_output.device)
            elif _SYNC_BN_V2 or _SYNC_BN_V3:
                # before 1.9.0 we need the count as an integer to compute means values
                count_all_sum = count_all.sum()
                mean_dy = sum_dy / count_all_sum
                mean_dy_xmu = sum_dy_xmu / count_all_sum
            else:
                # before 1.5.0, sum_dy was sum of means from every worker, so we just
                # need to divide it by number of workers
                mean_dy = sum_dy / size()
                mean_dy_xmu = sum_dy_xmu / size()

            # backward pass for gradient calculation
            # we are calling into a non-public undocumented function which broke moving to 1.9.0
            # https://github.com/pytorch/pytorch/issues/57900
            if _SYNC_BN_V4:
                # from 1.9.0 on, sums and count parameters expected
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    sum_dy,
                    sum_dy_xmu,
                    count_all
                )
            else:
                # before 1.9.0, mean parameters expected, not sums and count
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    mean_dy,
                    mean_dy_xmu
                )
        else:
            grad_input = None

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not need_weight_grad:
            grad_weight = None

        if weight is None or not need_bias_grad:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
