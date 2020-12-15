# Copyright 2020 Amazon Technologies, Inc. All Rights Reserved.
# Copyright 2019 ByteDance Technologies, Inc. All Rights Reserved.
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

import itertools
import unittest

import byteps.torch as bps
import torch
import numpy as np

from meta_test import MetaTest


class TorchTest(unittest.TestCase, metaclass=MetaTest):
    """
    Tests for ops in byteps.torch.
    """

    def test_byteps_push_pull(self):
        """Test that the byteps_push_pull correctly sums 1D, 2D, 3D tensors."""
        dtypes = [torch.float16, torch.float32, torch.float64]
        dims = [1, 2, 3]
        count = 0
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        shapes = [(), (17,), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            # MXNet uses gpu_id as part of the seed, so to get identical seeds
            # we must set a context.
            tensor = torch.randn(size=shapes[dim]).to(device)
            tensor = tensor.type(dtype)
            input = tensor.clone()

            bps.declare("tensor_" + str(count))
            handle = bps.byteps_push_pull(tensor, name="tensor_"+str(count))
            bps.synchronize(handle)
            assert torch.allclose(input, tensor)
            count += 1

        print('test_byteps_push_pull passed')

    def test_byteps_push_pull_temp_tensor(self):
        """Test pushpull in temporary tensors"""
        bps.declare("tmp")
        shape = (2,)
        handle = bps.byteps_push_pull(torch.empty(shape).cuda(), name="tmp")
        bps.synchronize(handle)
        for _ in range(10):
            x = torch.randn(shape)
            handle = bps.byteps_push_pull(x.clone().cuda(), name="tmp")
            output = bps.synchronize(handle)
            assert torch.allclose(output, x.cuda())

    def test_byteps_push_pull_fp16_nan(self):
        """Test robustness"""
        bps.declare("nan")
        bps.declare("good")

        nan_tensor = torch.rand(100).cuda() * 1e5
        nan_tensor = nan_tensor.type(torch.float16)
        handle = bps.byteps_push_pull(nan_tensor, average=False, name="nan")
        bps.synchronize(handle)

        good_tensor = torch.rand(100).cuda()
        good_tensor = good_tensor.type(torch.float16)
        input = good_tensor.cpu().numpy()
        handle = bps.byteps_push_pull(good_tensor, average=False, name="good")
        bps.synchronize(handle)
        output = good_tensor.cpu().numpy()

        assert np.allclose(input, output, np.finfo(np.float16).eps)
        print('test_byteps_push_pull_fp16_nan passed')


if __name__ == '__main__':
    unittest.main()
