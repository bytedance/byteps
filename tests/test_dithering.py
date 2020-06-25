import unittest

import byteps.mxnet as bps
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from gluoncv.model_zoo import get_model
from mxnet import gluon, autograd
from parameterized import parameterized
from tqdm import tqdm

from utils import fake_data


def round_next_pow2(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v


# partition: 'linear' or 'natural'
def dithering(x, k, partition='linear'):
    y = x.flatten()
    # normalize
    l2 = np.linalg.norm(y, ord=2)
    y /= l2

    # stocastic rounding
    if partition == 'linear':
        y *= k
        low = np.floor(y)
        p = y - low  # whether to ceil
        y = low + np.random.binomial(n=1, p=p)
        y /= k
    else:
        # 2 < 3 < 4
        y *= 2**(k-1)
        low = round_next_pow2(int(np.ceil(y))) << 1
        p = (y - low) / low
        y = (1 + np.random.binomial(n=1, p=p)) * low
        y /= 2**(k-1)

    return y.reshape(x.shape)


class DitheringTestCase(unittest.TestCase):
    def setUp(self):
        print("init")
        bps.init()

    @parameterized.expand([(1,)])
    def test_dithering(self, k):
        ctx = mx.gpu(0)
        net = get_model("resnet18_v2")
        net.initialize(mx.init.Xavier(), ctx=ctx)
        net.summary(nd.ones((1, 3, 224, 224), ctx=ctx))

        # hyper-params
        batch_size = 32
        optimizer_params = {'momentum': 0.9, 'wd': 1e-4,
                            'learning_rate': 0.01}

        compression_params = {
            "compressor": "dithering",
            "ef": "vanilla",
            "momentum": "nesterov",
            "k": k,
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
                    moms[i] *= 0.9
                    moms[i] += g
                    g += 0.9 * moms[i]
                    g += errors[i]
                    c = dithering(g, k)
                    errors[i] = g - c

                    # c += errors_s[i]
                    # cs = dithering(c, k)
                    # errors_s[i] = c - cs
                    # c = cs

                    c += 1e-4*xs[i]
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


if __name__ == '__main__':
    unittest.main()
