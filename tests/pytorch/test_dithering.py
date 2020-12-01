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

import copy
import itertools
import unittest

import byteps.torch as bps
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from numba import jit
from parameterized import parameterized
from tqdm import tqdm

from meta_test import MetaTest
from utils import bernoulli, fake_data


@jit(nopython=True)
def round_next_pow2(v):
    v -= np.uint32(1)
    v |= v >> np.uint32(1)
    v |= v >> np.uint32(2)
    v |= v >> np.uint32(4)
    v |= v >> np.uint32(8)
    v |= v >> np.uint32(16)
    v += np.uint32(1)
    return v


def dithering(x, k, state, partition='linear', norm="max"):
    dtype = x.dtype
    y = x.flatten().astype(np.float32)
    if norm == "max":
        scale = np.max(np.abs(y))
    elif norm == "l2":
        scale = np.linalg.norm(y, ord=2)
    else:
        raise ValueError("Unsupported normalization")
    sign = np.array(0 < y).astype(np.int32) - np.array(y < 0).astype(np.int32)
    y = np.abs(y)
    y /= scale

    # stocastic rounding
    if partition == 'linear':
        y *= k
        low = np.floor(y)
        p = y - low  # whether to ceil
        y = low + bernoulli(p, state)
        y *= scale
        y /= k
    elif partition == "natural":
        y *= 2**(k-1)
        low = round_next_pow2((np.ceil(y).astype(np.uint32))) >> 1
        length = copy.deepcopy(low)
        length[length == 0] = 1
        p = (y - low) / length
        y = low + length * bernoulli(p, state)
        y = y.astype(np.float32)
        y *= scale
        y /= 2**(k-1)
    else:
        raise ValueError("Unsupported partition")

    y *= sign
    return y.reshape(x.shape).astype(dtype)


class OnebitTestCase(unittest.TestCase, metaclass=MetaTest):
    TEST_BENCH = [
        [2, 4, 8],
        ["linear", "natural"],
        ["max", "l2"],
        [torch.float32, torch.float16],
        np.random.randint(0, 2020, size=3).tolist()
    ]

    @parameterized.expand(itertools.product(*TEST_BENCH))
    def test_dithering(self, k, ptype, ntype, dtype, seed):
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
            "compressor": "dithering",
            "k": k,
            "partition": ptype,
            "normalize": ntype,
            "seed": seed,
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
                    s = seed + i
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
                        c = dithering(g, k, rngs[param], ptype, ntype)

                        cs = dithering(c, k, rngs_s[param], ptype, ntype)
                        c = cs

                        params[param] -= lr * c

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
