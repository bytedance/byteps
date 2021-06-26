# Based on https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/_functions.py
# Modifications copyright 2020 Maka Autonomous Robotic Systems
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

from byteps.torch.ops import push_pull_async, allgather_async, size, rank, synchronize, local_size, local_rank

from distutils.version import LooseVersion

import os
import torch
import time
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from pprint import pprint


my_syncbn_dict = dict()
my_counter = 0

# Backward compat for old PyTorch
if not hasattr(torch.jit, 'unused'):
    torch.jit.unused = lambda x: x


_SYNC_BN_V2 = (
    LooseVersion(torch.__version__) >= LooseVersion('1.5.0') and
    LooseVersion(torch.__version__) <= LooseVersion('1.6.0')
)
_SYNC_BN_V3 = LooseVersion(torch.__version__) >= LooseVersion('1.6.0')


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
        global my_counter
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.mark = torch.tensor([my_counter])
        my_counter += 1
        node_local = os.getenv('BYTEPS_SYNC_BN_GLOBAL', 'False').lower() not in ["true", "1"]
        self.sync_size = size()
        if node_local:
            self.sync_size = local_size()
        self.skip_syncbn = os.getenv('BYTEPS_SKIP_SYNC_BN', 'False').lower() in ["true", "1"]

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'.format(input.dim()))

    def _run_bn(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

    @torch.jit.unused
    def _maybe_run_sync_bn(self, input):
        # if size() == 1:
        if self.sync_size == 1:
            return self._run_bn(input)
        return _SyncBatchNorm.apply(
            input, self.weight, self.bias, self.running_mean, self.running_var,
            self.eps, self.momentum, self.num_features, self.mark)

    def forward(self, input):
        # currently only GPU input is supported by underlying kernel from PyTorch
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1

        if not self.training and self.track_running_stats:
            return self._run_bn(input)
        elif self.skip_syncbn:
            return self._run_bn(input)
        else:
            return self._maybe_run_sync_bn(input)


class _SyncBatchNorm(Function):
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, num_features, mark):
        global my_syncbn_dict
        node_local = os.getenv('BYTEPS_SYNC_BN_GLOBAL', 'False').lower() not in ["true", "1"]
        my_rank = rank()
        sync_size = size()
        if node_local:
            my_rank = local_rank()
            sync_size = local_size()
        input = input.contiguous()
        weight = weight.contiguous()

        my_size = input.numel() // input.size(1)
        tmp_size = [0] * sync_size
        tmp_size[my_rank] = my_size
        count = torch.tensor(tmp_size).cuda()

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)
        raw_mean = mean
        raw_invstd = invstd

        my_unique_id = mark.numpy()[0]
        if my_unique_id not in my_syncbn_dict:
            my_syncbn_dict[my_unique_id] = {
               "tmp_count": torch.zeros((sync_size,), dtype=mean.dtype, device=mean.device),
               "tmp_mean": torch.zeros((sync_size, num_features), dtype=mean.dtype, device=mean.device),
               "tmp_invstd": torch.zeros((sync_size, num_features), dtype=mean.dtype, device=mean.device),
               "tmp_sum_dy": torch.zeros((num_features,), dtype=mean.dtype, device=mean.device),
               "tmp_sum_dy_xmu": torch.zeros((num_features,), dtype=mean.dtype, device=mean.device),
               }
            tmp_dict = my_syncbn_dict[my_unique_id]
            tmp_count = tmp_dict["tmp_count"]
            tmp_mean = tmp_dict["tmp_mean"]
            tmp_invstd = tmp_dict["tmp_invstd"]
            tmp_sum_dy = tmp_dict["tmp_sum_dy"]
            tmp_sum_dy_xmu = tmp_dict["tmp_sum_dy_xmu"]
            tmp_dict["name_count"] = 'sync_batch_norm.count.' + str(torch.numel(tmp_count)) + '.' + str(tmp_count.dtype) + '.' + str(my_unique_id)
            tmp_dict["name_mean"] = 'sync_batch_norm.mean.' + str(torch.numel(tmp_mean)) + '.' + str(tmp_mean.dtype) + '.' + str(my_unique_id)
            tmp_dict["name_invstd"] = 'sync_batch_norm.invstd.' + str(torch.numel(tmp_invstd)) + '.' + str(tmp_invstd.dtype) + '.' + str(my_unique_id)
            tmp_dict["name_sum_dy"] = 'sync_batch_norm.sum_dy.' + str(torch.numel(tmp_sum_dy)) + '.' + str(tmp_sum_dy.dtype) + '.' + str(my_unique_id)
            tmp_dict["name_sum_dy_xmu"] = 'sync_batch_norm.sum_dy_xmu.' + str(torch.numel(tmp_sum_dy_xmu)) + '.' + str(tmp_sum_dy_xmu.dtype) + '.' + str(my_unique_id)
        tmp_dict = my_syncbn_dict[my_unique_id]
        tmp_count = tmp_dict["tmp_count"].fill_(0)
        tmp_mean = tmp_dict["tmp_mean"].fill_(0)
        tmp_invstd = tmp_dict["tmp_invstd"].fill_(0)
        name_count = tmp_dict["name_count"]
        name_mean = tmp_dict["name_mean"]
        name_invstd = tmp_dict["name_invstd"]

        tmp_invstd[my_rank].copy_(raw_invstd)
        tmp_mean[my_rank].copy_(raw_mean)
        tmp_count[my_rank] = my_size

        # raw_count = torch.tensor([my_size*1.0]).cuda()
        raw_count = torch.full((1,), input.numel() // input.size(1),
                               dtype=mean.dtype,
                               device=mean.device)
        count_handle = allgather_async(raw_count.unsqueeze(0), name=name_count, node_local=node_local)
        mean_handle = allgather_async(raw_mean.unsqueeze(0), name=name_mean, node_local=node_local)
        invstd_handle = allgather_async(raw_invstd.unsqueeze(0), name=name_invstd, node_local=node_local)

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

        self.save_for_backward(input, weight, mean, invstd, count_all.to(torch.int32), mark)

        # apply element-wise normalization
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_all, mark = self.saved_tensors
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
            node_local = os.getenv('BYTEPS_SYNC_BN_GLOBAL', 'False').lower() not in ["true", "1"]
            my_rank = rank()
            sync_size = size()
            if node_local:
                my_rank = local_rank()
                sync_size = local_size()

            my_unique_id = mark.numpy()[0]
            assert  my_unique_id in my_syncbn_dict
            tmp_dict = my_syncbn_dict[my_unique_id]

            tmp_sum_dy = tmp_dict["tmp_sum_dy"]
            tmp_sum_dy_xmu = tmp_dict["tmp_sum_dy_xmu"]
            name_sum_dy = tmp_dict["name_sum_dy"]
            name_sum_dy_xmu = tmp_dict["name_sum_dy_xmu"]

            tmp_sum_dy.copy_(sum_dy)
            tmp_sum_dy_xmu.copy_(sum_dy_xmu)
            sum_dy = tmp_sum_dy
            sum_dy_xmu = tmp_sum_dy_xmu

            sum_dy_handle = push_pull_async(sum_dy, average=False, name=name_sum_dy, node_local=node_local)
            sum_dy_xmu_handle = push_pull_async(sum_dy_xmu, average=False, name=name_sum_dy_xmu,  node_local=node_local)

            # wait on the async communication to finish
            sum_dy = synchronize(sum_dy_handle)
            sum_dy_xmu = synchronize(sum_dy_xmu_handle)

            if _SYNC_BN_V2 or _SYNC_BN_V3:
                count_all_sum = count_all.sum()
                mean_dy = sum_dy / count_all_sum
                mean_dy_xmu = sum_dy_xmu / count_all_sum
            else:
                # before 1.5.0, sum_dy was sum of means from every worker, so we just
                # need to divide it by number of workers
                mean_dy = sum_dy / sync_size
                mean_dy_xmu = sum_dy_xmu / sync_size

            # backward pass for gradient calculation
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
