import unittest

import byteps.mxnet as bps
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from gluoncv.model_zoo import get_model
from mxnet import gluon, autograd
from parameterized import parameterized
from tqdm import tqdm

from utils import fake_data, XorShift128PlusBitShifterRNG


def randomk(x, k, rng):
    y = x.flatten()
    indices = [rng.randint(0, len(y)) for _ in range(k)]
    vals = y[indices]
    y.fill(0)
    for idx, val in zip(indices, vals):
        y[idx] = val
    return y.reshape(x.shape)


class RandomkTestCase(unittest.TestCase):
    def setUp(self):
        print("init")
        bps.init()

    @parameterized.expand([(1,)])
    def test_randomk(self, k):
        ctx = mx.gpu(0)
        # np.random.seed(2020)
        net = get_model("resnet18_v2")
        net.initialize(mx.init.Xavier(), ctx=ctx)
        net.summary(nd.ones((1, 3, 224, 224), ctx=ctx))

        # hyper-params
        batch_size = 32
        optimizer_params = {'momentum': 0, 'wd': 0,
                            'learning_rate': 0.01}

        compression_params = {
            "compressor": "randomk",
            # "ef": "vanilla",
            # "momentum": "nesterov",
            "k": k,
            "seed": 2020
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
        rngs = {}
        rngs_s = {}

        for i, param in enumerate(trainer._params):
            if param.grad_req != 'null':
                params[i] = param._data[0].asnumpy()
                errors[i] = np.zeros_like(params[i])
                errors_s[i] = np.zeros_like(params[i])
                moms[i] = np.zeros_like(params[i])
                wd_moms[i] = np.zeros_like(params[i])
                rngs[i] = XorShift128PlusBitShifterRNG(2020, 2020)
                rngs_s[i] = XorShift128PlusBitShifterRNG(2020, 2020)

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
                    c = randomk(g, k, rngs[i])
                    # errors[i] = g - c

                    # c += errors_s[i]
                    cs = randomk(c, k, rngs_s[i])
                    # errors_s[i] = c - cs
                    c = cs

                    # c += 1e-4*xs[i]
                    params[i] -= optimizer_params["learning_rate"] * c

                    g2 = param._grad[0].asnumpy().flatten()
                    d = c.flatten()
                    if not np.allclose(d, g2, atol=np.finfo(np.float32).eps):
                        print("False")

                        diff = np.abs(d - g2)
                        print(d) # baseline
                        print(g2) # byteps
                        print(diff)
                        print(it, i, np.max(diff), np.mean(diff), len(diff), c.shape)
                        idx = np.where(diff > 1e-5)
                        print("g: ", idx, gs[i].flatten()[idx])
                        print("g+e: ", idx, g.flatten()[idx])
                        print("mxnet: ", idx, d[idx])
                        print("byteps: ", idx, g2[idx])
                        input()

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
