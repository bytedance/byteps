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

import byteps.mxnet as bps
import mxnet as mx
import numpy as np

from meta_test import MetaTest

has_gpu = mx.context.num_gpus() > 0


class MXTest(unittest.TestCase, metaclass=MetaTest):
    """
    Tests for ops in byteps.mxnet.
    """
    def _current_context(self):
        if has_gpu:
            return mx.gpu(bps.local_rank())
        else:
            return mx.current_context()
    
    def test_byteps_trainer_param_order(self):
        net = mx.gluon.nn.Sequential()
        # layers may be added in a random order for all workers
        layers = {'ones_': 1, 'zeros_': 0}
        for name, init in layers.items():
            net.add(mx.gluon.nn.Dense(10, in_units=10, weight_initializer=mx.init.Constant(init),
                                      use_bias=False, prefix=name))
        params = net.collect_params()
        net.initialize()
        trainer = bps.DistributedTrainer(params, 'sgd')
        trainer._init_params()
        # check the result of bps_broadcast
        for name, init in layers.items():
            weight = params[name + 'weight'].data()[0].asnumpy()
            expected = np.full(shape=weight.shape,
                               fill_value=init, dtype=weight.dtype)
            assert np.array_equal(weight, expected), (weight, expected)

        print('test_byteps_trainer_param_order passed')

    def test_byteps_push_pull(self):
        """Test that the byteps_push_pull correctly sums 1D, 2D, 3D tensors."""
        dtypes = ['float16', 'float32', 'float64']
        dims = [1, 2, 3]
        count = 0
        ctx = self._current_context()
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            # MXNet uses gpu_id as part of the seed, so to get identical seeds
            # we must set a context.
            mx.random.seed(10 + 10 * bps.rank(), ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            input = tensor.asnumpy()

            bps.byteps_declare_tensor("tensor_" + str(count))
            bps.byteps_push_pull(tensor, name="tensor_"+str(count))
            tensor.wait_to_read()
            output = tensor.asnumpy()
            assert np.allclose(input, output)
            count += 1

        print('test_byteps_push_pull passed')

    def test_byteps_push_pull_inplace(self):
        """Test that the byteps_push_pull correctly sums 1D, 2D, 3D tensors."""
        size = bps.size()
        dtypes = ['float16', 'float32', 'float64']
        dims = [1, 2, 3]
        count = 0
        ctx = self._current_context()
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            multiplied = tensor.copy()
            bps.byteps_declare_tensor("tensor_" + str(count))
            bps.byteps_push_pull(tensor, name="tensor_" + str(count))
            max_difference = mx.nd.max(mx.nd.subtract(tensor, multiplied))
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            if max_difference > threshold:
                print("self", count, dtype, dim, max_difference, threshold)
                print("tensor", bps.rank(), tensor)
                print("multiplied", bps.rank(), multiplied)
            assert max_difference <= threshold, 'bps.byteps_push_pull produces \
                                                 incorrect results for self'

        print('test_byteps_push_pull_inplace passed')


if __name__ == '__main__':
    unittest.main()
