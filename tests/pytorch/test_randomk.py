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
from utils import fake_data, randint


def randomk(x, k, state):
    y = x.flatten()
    low = np.uint64(0)
    high = np.uint64(len(y))
    indices = np.array([randint(low, high, state)
                        for _ in range(k)], dtype=np.uint64)
    vals = y[indices]
    y.fill(0)
    scale = len(y) / k
    for idx, val in zip(indices, vals):
        y[idx] = val * scale
    return y.reshape(x.shape)


class RandomkTestCase(unittest.TestCase, metaclass=MetaTest):
    TEST_BENCH = [
        [3],
        [torch.float32, torch.float16],
        np.random.randint(0, 2020, size=1).tolist()
    ]

    @parameterized.expand(itertools.product(*TEST_BENCH))
    def test_randomk(self, k, dtype, seed):
        bps.init()
        np_dtype = str(dtype).split('.')[1]

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = torchvision.models.resnet18().to(device)
        net = net.type(dtype)

        # hyper-params
        batch_size = 32
        lr = 0.01
        compression_params = {
            "compressor": "randomk",
            "k": k,
            "seed": seed
        }

        optimizer = optim.SGD(net.parameters(), lr=lr)
        optimizer = bps.DistributedOptimizer(
            optimizer, named_parameters=net.named_parameters(),
            compression_params=compression_params)

        train_data = fake_data(batch_size=batch_size)

        params = {}
        rngs = {}
        rngs_s = {}

        for i, param_group in enumerate(optimizer.param_groups):
            for param in param_group['params']:
                if param.requires_grad:
                    params[param] = param.data.cpu().numpy()
                    s = seed + i + k
                    rngs[param] = np.array([s, s], dtype=np.uint64)
                    rngs_s[param] = np.array([s, s], dtype=np.uint64)

        for it, batch in tqdm(enumerate(train_data)):
            data = batch[0].to(device).type(dtype)
            label = batch[1].to(device)

            output = net(data)
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()

            gs = {}
            xs = {}

            for i, param_group in enumerate(optimizer.param_groups):
                for param in param_group['params']:
                    if param.requires_grad:
                        gs[param] = param.grad.cpu().numpy()
                        xs[param] = param.data.cpu().numpy()

            optimizer.step()

            for i, param_group in enumerate(optimizer.param_groups):
                for param in param_group['params']:
                    if param.requires_grad:
                        g = gs[param] / bps.size()
                        print("g", g)
                        c = randomk(g, k, rngs[param])

                        params[param] -= lr * c
                        np_g = c.flatten()
                        th_g = param.grad.cpu().numpy().flatten()
                        if not np.allclose(np_g, th_g, atol=np.finfo(np_dtype).eps):
                            diff = np.abs(np_g - th_g)
                            # print("g", g.flatten())
                            print("np", np_g)
                            print("th", th_g)
                            print("diff", diff)
                            print("max diff", np.max(diff))
                            print("len", len(diff))
                            idx = np.nonzero(diff > np.finfo(np_dtype).eps)
                            print("idx", idx, np_g[idx],
                                  th_g[idx], g.flatten()[idx])
                            input()

        cnt = 0
        tot = 0
        threshold = 0 if dtype == torch.float32 else 10
        for i, param_group in enumerate(optimizer.param_groups):
            for param in param_group['params']:
                if param.requires_grad:
                    x = param.data.cpu().numpy()
                    tot += len(x.flatten())
                    if not np.allclose(params[param], x, atol=np.finfo(np_dtype).eps):
                        diff = np.abs(x.flatten() - params[param].flatten())
                        idx = np.where(diff > np.finfo(np_dtype).eps)
                        cnt += len(idx[0])

        assert cnt <= threshold, "false/tot=%d/%d=%f" % (
            cnt, tot, cnt/tot)


if __name__ == '__main__':
    unittest.main()
