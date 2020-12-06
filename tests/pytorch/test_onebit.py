# Copyright 2020 Amazon Technologies, Inc. All Rights Reserved.
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
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from parameterized import parameterized
from tqdm import tqdm

from meta_test import MetaTest
from utils import fake_data


def onebit(x, scaling):
    if scaling:
        l1 = np.linalg.norm(x.flatten(), 1)
    sign = x < 0
    sign = -((sign << 1) - 1)
    if scaling:
        return l1 / len(x.flatten()) * sign
    else:
        return sign


class OnebitTestCase(unittest.TestCase, metaclass=MetaTest):
    # TEST_BENCH = [
    #     [True, False],
    #     [torch.float32, torch.float16]
    # ]

    # @parameterized.expand(itertools.product(*TEST_BENCH))
    # def test_onebit(self, scaling, dtype):
    #     bps.init()
    #     np_dtype = str(dtype).split('.')[1]

    #     device = torch.device(
    #         "cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     net = torchvision.models.resnet18().to(device)
    #     net = net.type(dtype)

    #     # hyper-params
    #     batch_size = 32
    #     lr = 0.01
    #     compression_params = {
    #         "compressor": "onebit",
    #         "scaling": scaling
    #     }

    #     optimizer = optim.SGD(net.parameters(), lr=lr)
    #     optimizer = bps.DistributedOptimizer(
    #         optimizer, named_parameters=net.named_parameters(),
    #         compression_params=compression_params)

    #     train_data = fake_data(batch_size=batch_size)

    #     params = {}

    #     for i, param_group in enumerate(optimizer.param_groups):
    #         for param in param_group['params']:
    #             if param.requires_grad:
    #                 params[param] = param.data.cpu().numpy()

    #     for it, batch in tqdm(enumerate(train_data)):
    #         data = batch[0].to(device).type(dtype)
    #         label = batch[1].to(device)

    #         output = net(data)
    #         loss = F.cross_entropy(output, label)
    #         optimizer.zero_grad()
    #         loss.backward()

    #         gs = {}
    #         xs = {}

    #         for i, param_group in enumerate(optimizer.param_groups):
    #             for param in param_group['params']:
    #                 if param.requires_grad:
    #                     gs[param] = param.grad.cpu().numpy()
    #                     xs[param] = param.data.cpu().numpy()

    #         optimizer.step()

    #         for i, param_group in enumerate(optimizer.param_groups):
    #             for param in param_group['params']:
    #                 if param.requires_grad:
    #                     g = gs[param] / bps.size()
    #                     c = onebit(g, scaling)

    #                     cs = onebit(c, scaling)
    #                     c = cs

    #                     params[param] -= lr * c

    #     cnt = 0
    #     tot = 0
    #     threshold = 0 if dtype == torch.float32 else 10
    #     for i, param_group in enumerate(optimizer.param_groups):
    #         for param in param_group['params']:
    #             if param.requires_grad:
    #                 x = param.data.cpu().numpy()
    #                 tot += len(x.flatten())
    #                 if not np.allclose(params[param], x, atol=np.finfo(np_dtype).eps):
    #                     diff = np.abs(x.flatten() - params[param].flatten())
    #                     idx = np.where(diff > np.finfo(np_dtype).eps)
    #                     cnt += len(idx[0])

    #     assert cnt <= threshold, "false/tot=%d/%d=%f" % (
    #         cnt, tot, cnt/tot)

    def test_byteps_push_pull_fp16_nan(self):
        """
        """
        for i in range(10):
            tensor = torch.rand(100).cuda() * 1e5
            tensor = tensor.type(torch.float16)
            input = tensor.cpu().numpy().astype(np.float32)
            input = np.nan_to_num(input, nan=0, posinf=0,
                                  neginf=0)
            input = onebit(input, True)
            print(input)
            bps.declare("tensor_" + str(i), **{
                "byteps_compressor_type": "onebit",
                "byteps_compressor_onebit_scaling": "true"
            })
            handle = bps.byteps_push_pull(tensor, name="tensor_" + str(i))
            output = bps.synchronize(handle)
            output = tensor.cpu().numpy()

            print(output)
            assert np.allclose(input, output, np.finfo(np.float16).eps)
        print('test_byteps_push_pull_fp16_nan passed')


if __name__ == '__main__':
    unittest.main()
