import unittest

import byteps.mxnet as bps
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from gluoncv.model_zoo import get_model
from mxnet import autograd, gluon
from tqdm import tqdm

from utils import fake_data


def onebit(x):
    l1 = np.linalg.norm(x.flatten(), 1)
    sign = x < 0
    sign = -((sign << 1) - 1)
    return l1 / len(x.flatten()) * sign


class OnebitTestCase(unittest.TestCase):
    def test_onebit(self):
        bps.init()
        ctx = mx.gpu(0)
        net = get_model("resnet18_v2")
        net.initialize(mx.init.Xavier(), ctx=ctx)
        net.summary(nd.ones((1, 3, 224, 224), ctx=ctx))

        # hyper-params
        batch_size = 32
        optimizer_params = {'momentum': 0, 'wd': 0,
                            'learning_rate': 0.01}

        compression_params = {
            "compressor": "onebit",
            # "ef": "vanilla",
            # "momentum": "nesterov",
            "scaling": True,
        }

        trainer = bps.DistributedTrainer(net.collect_params(
        ), "sgd", optimizer_params, compression_params=compression_params)

        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        train_data = fake_data(batch_size=batch_size)

        params = {}
        errors = {}
        errors_s = {}
        moms = {}
        wd_moms = {}

        for i, param in enumerate(trainer._params):
            if param.grad_req != 'null':
                params[i] = param._data[0].asnumpy()
                errors[i] = np.zeros_like(params[i])
                errors_s[i] = np.zeros_like(params[i])
                moms[i] = np.zeros_like(params[i])
                wd_moms[i] = np.zeros_like(params[i])

        for it, batch in tqdm(enumerate(train_data)):
            data = batch[0].as_in_context(ctx)
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
                    # moms[i] *= 0.9
                    # moms[i] += g
                    # g += 0.9 * moms[i]
                    # g += errors[i]
                    c = onebit(g)
                    # errors[i] = g - c

                    # c += errors_s[i]
                    cs = onebit(c)
                    # errors_s[i] = c - cs
                    c = cs

                    # wd_moms[i] = 0.9 * wd_moms[i] + 1e-4 * xs[i]
                    # c += 0.9 * wd_moms[i] + 1e-4 * xs[i]
                    params[i] -= optimizer_params["learning_rate"] * c

        cnt = 0
        tot = 0
        diffs = []
        for i, param in enumerate(trainer._params):
            if param.grad_req != "null":
                x = param._data[0].asnumpy()
                tot += len(x.flatten())
                if not np.allclose(params[i], x, atol=np.finfo(np.float32).eps):
                    diff = np.abs(x.flatten() - params[i].flatten())
                    diffs.append(np.max(diff))
                    idx = np.where(diff > np.finfo(np.float32).eps)
                    cnt += len(idx[0])

        print("false=%d tot=%d false / tot = %lf" % (cnt, tot, cnt / tot))
        if diffs:
            print("max_diff=%f\tmin_diff=%f\tmean_diff=%f" %
                  (np.max(diffs), np.min(diffs), np.mean(diffs)))

        assert cnt == 0


if __name__ == '__main__':
    unittest.main()
