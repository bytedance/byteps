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

import byteps.mxnet as bps
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from gluoncv.model_zoo import get_model
from mxnet import autograd, gluon
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
    TEST_BENCH = [
        [True, False],
        ["float32", "float16"]
    ]

    @parameterized.expand(itertools.product(*TEST_BENCH))
    def test_onebit(self, scaling, dtype):
        bps.init()
        ctx = mx.gpu(0)
        net = get_model("resnet18_v2")
        net.cast(dtype)
        net.initialize(mx.init.Xavier(), ctx=ctx)
        net.summary(nd.ones((1, 3, 224, 224),
                            ctx=ctx).astype(dtype, copy=False))

        # hyper-params
        batch_size = 32
        optimizer_params = {'momentum': 0, 'wd': 0,
                            'learning_rate': 0.01}

        compression_params = {
            "compressor": "onebit",
            "scaling": scaling,
            "fp16": True if dtype == "float16" else False
        }

        trainer = bps.DistributedTrainer(net.collect_params(
        ), "sgd", optimizer_params, compression_params=compression_params)

        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        train_data = fake_data(batch_size=batch_size)

        params = {}

        for i, param in enumerate(trainer._params):
            if param.grad_req != 'null':
                params[i] = param._data[0].asnumpy()

        for it, batch in tqdm(enumerate(train_data)):
            data = batch[0].as_in_context(ctx).astype(dtype, copy=False)
            label = batch[1].as_in_context(ctx)

            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)

            loss.backward()

            gs = {}
            xs = {}

            for i, param in enumerate(trainer._params):
                if param.grad_req != 'null':
                    gs[i] = param._grad[0].asnumpy()
                    xs[i] = param._data[0].asnumpy()

            trainer.step(batch_size)

            for i, param in enumerate(trainer._params):
                if param.grad_req != "null":
                    g = gs[i] / (batch_size * bps.size())
                    c = onebit(g, scaling)

                    cs = onebit(c, scaling)
                    c = cs

                    params[i] -= optimizer_params["learning_rate"] * c

        cnt = 0
        tot = 0
        threshold = 0 if dtype == "float32" else 10
        for i, param in enumerate(trainer._params):
            if param.grad_req != "null":
                x = param._data[0].asnumpy()
                tot += len(x.flatten())
                if not np.allclose(params[i], x, atol=np.finfo(dtype).eps):
                    diff = np.abs(x.flatten() - params[i].flatten())
                    idx = np.where(diff > np.finfo(dtype).eps)
                    cnt += len(idx[0])

        assert cnt <= threshold, "false/tot=%d/%d=%f" % (
            cnt, tot, cnt/tot)

    def test_byteps_push_pull_fp16_nan(self):
        """
        """
        for i in range(10):
            tensor = mx.nd.random.uniform(-1e5, 1e5, shape=100, ctx=mx.gpu(0))
            tensor = tensor.astype('float16')
            input = tensor.asnumpy().astype(np.float32)
            input = np.nan_to_num(input, nan=0, posinf=0,
                                  neginf=0)
            input = onebit(input, True)
            print(input)
            bps.byteps_declare_tensor("tensor_" + str(i), **{
                "byteps_compressor_type": "onebit",
                "byteps_compressor_onebit_scaling": "true"
            })
            bps.byteps_push_pull(tensor, name="tensor_" + str(i))
            tensor.wait_to_read()
            output = tensor.asnumpy()

            print(output)
            assert np.allclose(input, output, np.finfo(np.float16).eps)
        print('test_byteps_push_pull_fp16_nan passed')


if __name__ == '__main__':
    unittest.main()
