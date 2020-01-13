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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import byteps.mxnet as bps
import itertools
import mxnet as mx
import os
import numpy as np
import unittest
from mxnet.base import MXNetError
from mxnet.test_utils import same

has_gpu = mx.context.num_gpus() > 0

# MLSL supports only byte, float and double data types
mlsl_supported_types = set(['float32', 'float64'])

class MXTest:
    """
    Tests for ops in byteps.mxnet.
    """

    def _current_context(self):
        if has_gpu:
            return mx.gpu(bps.local_rank())
        else:
            return mx.current_context()

    def filter_supported_types(self, types):
        if 'MLSL_ROOT' in os.environ:
           types = [t for t in types if t in mlsl_supported_types]
        return types

    def test_byteps_trainer_param_order(self):
        size = bps.size()
        dtypes = self.filter_supported_types(['float32'])
        dims = [1]
        ctx = self._current_context()
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
            expected = np.full(shape=weight.shape, fill_value=init, dtype=weight.dtype)
            assert np.array_equal(weight, expected), (weight, expected)

        print('test_byteps_trainer_param_order passed')

    def test_byteps_push_pull(self):
        """Test that the byteps_push_pull correctly sums 1D, 2D, 3D tensors."""
        size = bps.size()
        dtypes = self.filter_supported_types(['float32'])
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(), (17)]
        for dtype, dim in itertools.product(dtypes, dims):
            # MXNet uses gpu_id as part of the seed, so to get identical seeds
            # we must set a context.
            mx.random.seed(10 + 10 * bps.rank(), ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)

            print("tensor before push_pull:", tensor)
            bps.byteps_declare_tensor("tensor_" + str(count))
            bps.byteps_push_pull(tensor, name="tensor_"+str(count))
            tensor.wait_to_read()
            print("tensor after push_pull:", tensor)

        print('test_byteps_push_pull passed')


    def test_byteps_push_pull_inplace(self):
        """Test that the byteps_push_pull correctly sums 1D, 2D, 3D tensors."""
        size = bps.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float32', 'float64'])
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 200
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            multiplied = tensor.copy()
            bps.byteps_declare_tensor("tensor_" + str(count))
            bps.byteps_push_pull(tensor, name= "tensor_" + str(count))
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


    def test_byteps_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        rank = bps.rank()
        size = bps.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 300
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims,
                                                       root_ranks):
            tensor = mx.nd.ones(shapes[dim], ctx=ctx) * rank
            root_tensor = mx.nd.ones(shapes[dim], ctx=ctx) * root_rank
            tensor = tensor.astype(dtype)
            root_tensor = root_tensor.astype(dtype)

            broadcast_tensor = bps.broadcast(tensor, root_rank=root_rank,
                                             name=str(count))
            if rank != root_rank:
                if same(tensor.asnumpy(), root_tensor.asnumpy()):
                    print("broadcast", count, dtype, dim,
                          mx.nd.max(tensor == root_tensor))
                    print("tensor", bps.rank(), tensor)
                    print("root_tensor", bps.rank(), root_tensor)
                    print("comparison", bps.rank(), tensor == root_tensor)
                assert not same(tensor.asnumpy(), root_tensor.asnumpy()), \
                    'bps.broadcast modifies source tensor'
            if not same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()):
                print("broadcast", count, dtype, dim)
                print("broadcast_tensor", bps.rank(), broadcast_tensor)
                print("root_tensor", bps.rank(), root_tensor)
                print("comparison", bps.rank(),
                      broadcast_tensor == root_tensor)
            assert same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()), \
                'bps.broadcast produces incorrect broadcasted tensor'


if __name__ == '__main__':
    mxtest = MXTest()
    bps.init()
    mxtest.test_byteps_push_pull()
    mxtest.test_byteps_trainer_param_order()
    #mxtest.test_byteps_broadcast()
    mxtest.test_byteps_push_pull_inplace()
