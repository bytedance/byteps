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

import unittest

import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon.model_zoo import vision

import byteps.mxnet as bps

from .datasets import fake_data
from .test_meta import TestMeta


def worker(model, input_data, dtype, config, compress=False, cpr_config=None):
    if model is None:
        raise ValueError("model is None")

    bps.init()

    ctx = mx.cpu(bps.local_rank()) if config.no_cuda else mx.gpu(
        bps.local_rank())
    num_workers = bps.size()

    train_data = input_data(config, ctx, dtype)

    model.cast(dtype)
    model.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    model.hybridize()

    params = model.collect_params()

    if compress and not cpr_config:
        for name, param in params.items():
            for name, value in cpr_config.items():
                setattr(param, name, value)

    optimizer_params = {'momentum': config.momentum,
                        'learning_rate': config.lr * num_workers}
    trainer = bps.DistributedTrainer(params, "sgd", optimizer_params)

    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()

    for epoch in range(config.epochs):
        metric.reset()
        for batch in train_data:
            data = batch[0].as_in_context(ctx)
            label = batch[1].as_in_context(ctx)

            with autograd.record():
                output = model(data)
                loss = loss_fn(output, label)

            loss.backward()
            trainer.step(config.batch_size)
            metric.update([label], [output])

    _, acc = metric.get()
    return acc


class OnebitCaseBase(unittest.TestCase, metaclass=TestMeta):
    def _model(self):
        return None

    def _config(self):
        return {
            "seed": 2020,
            "batch_size": 64,
            "data_size": 512,
            "epochs": 2,
            "num_workers": 2,
            "lr": 0.01,
            "momentum": 0.9,
            "no_cuda": True
        }

    def _cpr_config(self):
        return {
            "byteps_compressor_type", "onebit"
        }

    def _run(self, dtype):
        expected = worker(self._model, fake_data, dtype,
                          self._config, compress=False, self._cpr_config)
        actual = worker(self._model, fake_data, dtype,
                        self._config, compress=True, self._cpr_config)

        self.assertAlmostEqual(expected, actual)


class OnebitAlexnet(OnebitCaseBase):
    def _model(self):
        return vision.alexnet()


class OnebitVGG11(OnebitCaseBase):
    def _model(self):
        return vision.vgg11()


class OnebitResnet18(OnebitCaseBase):
    def _model(self):
        return vision.resnet18_v2()


del OnebitCaseBase


if __name__ == "__main__":
    unittest.main()
