# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2019 Intel Corporation
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

from distutils.version import LooseVersion

import inspect
import itertools
import os
import platform
import sys
import unittest
import warnings
import time
import json

from collections.abc import Iterable

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import byteps.torch as bps

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))


_1_5_api = LooseVersion(torch.__version__) >= LooseVersion('1.5.0')

ccl_supported_types = set([torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                           torch.IntTensor, torch.LongTensor, torch.FloatTensor,
                           torch.DoubleTensor])


class TorchTests(unittest.TestCase):
    """
    Tests for ops in horovod.torch.
    """

    def __init__(self, *args, **kwargs):
        super(TorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def convert_cpu_fp16_to_fp32(self, *values):
        # PyTorch doesn't support any CPU ops on FP16 tensors.
        # In case we need to do ops, we will convert tensor to FP32 here.
        result = []
        for value in values:
            if value.dtype in [torch.float16, torch.HalfTensor] and not value.is_cuda:
                result.append(value.float())
            else:
                result.append(value)
        return result

    def cast_and_place(self, tensor, dtype):
        if dtype.is_cuda:
            return tensor.cuda(bps.local_rank()).type(dtype)
        return tensor.type(dtype)

    def filter_supported_types(self, types):
        if 'CCL_ROOT' in os.environ:
           types = [t for t in types if t in ccl_supported_types]
        return types

    def test_gpu_required(self):
        if not torch.cuda.is_available():
            skip_or_fail_gpu_test(self, "No GPUs available")

    def test_horovod_sync_batch_norm(self):
        """Tests Horovod version of SyncBatchNorm."""
        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        bps.init()
        torch.cuda.set_device(bps.local_rank())

        ts_list = [
            torch.stack([
                torch.tensor([
                    [r, r + 1],
                    [r * 2, r * 2 + 1],
                    [r * 3, r * 3 + 1],
                    [r * 4, r * 4 + 1]
                ])
                for r in range(bps.size())
            ]),
            torch.stack([
                torch.tensor([
                    [r + 1],
                    [r * 2 + 1],
                    [r * 3 + 1],
                    [r * 4 + 1]
                ])
                for r in range(bps.size())
            ]),
        ]

        sync_bn = bps.SyncBatchNorm(num_features=4)
        sync_bn.cuda(bps.local_rank())

        bn = torch.nn.BatchNorm1d(num_features=4)
        bn.cuda(bps.local_rank())
        for idx, ts in enumerate(ts_list):

            ts = ts.cuda(bps.local_rank()).float()
            ts1 = ts.clone().requires_grad_()
            ts2 = ts.clone().requires_grad_()

            # Training
            sync_bn_out = sync_bn(ts1[bps.rank()].unsqueeze(0))
            bn_out = bn(ts2)
            assert torch.allclose(sync_bn_out, bn_out[bps.rank()].unsqueeze(0), 1e-6)
            assert torch.allclose(sync_bn.running_mean, bn.running_mean, 1e-6)
            assert torch.allclose(sync_bn.running_var, bn.running_var, 1e-6)

            # Gradients
            sync_bn_out.sum().backward()
            bn_out.mean(dim=0).sum().backward()
            assert torch.allclose(bps.push_pull(sync_bn.weight.grad, name='sync_bn.weight.grad.' + str(idx)), bn.weight.grad,  1e-6)
            assert torch.allclose(bps.push_pull(sync_bn.bias.grad, name='sync_bn.bias.grad.' + str(idx)), bn.bias.grad, 1e-6)
            assert torch.allclose(bps.push_pull(ts1.grad, name='ts1.grad.' + str(idx)), ts2.grad, 1e-6)
            break



if __name__ == "__main__":
   unittest.main()
