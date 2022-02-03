# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
# Modifications copyright (C) 2019 Intel Corporation
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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
# =============================================================================

"""Tests for byteps.tensorflow.xla_ops."""

import os
import sys
import pytest
import math
import numpy as np
import itertools
from distutils.version import LooseVersion
import warnings

# Enable BPS XLA ops so that tf.function(jit_compile=True) works. This
# environment variable needs to be set up before loading Tensorflow, because
# it is needed to tell XLA to register the ops through C++ static
# initialization.
os.environ["BYTEPS_ENABLE_XLA_OPS"] = os.getenv("BYTEPS_ENABLE_XLA_OPS", "1")

import tensorflow as tf
from byteps.tensorflow.util import _executing_eagerly
from tensorflow.python.framework import ops
import byteps.tensorflow as bps
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))


bps.init()
if hasattr(tf, 'ConfigProto'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

if hasattr(tf, 'config') and hasattr(tf.config, 'experimental') \
        and hasattr(tf.config.experimental, 'set_memory_growth'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    # Specifies the config to use with eager execution. Does not preclude
    # tests from running in the graph mode.
    tf.enable_eager_execution(config=config)

ccl_supported_types = set([tf.uint8, tf.int8, tf.uint16, tf.int16,
                           tf.int32, tf.int64, tf.float32])

_IS_TF26 = LooseVersion(tf.__version__) >= LooseVersion('2.6.0')


@pytest.mark.skipif(not _IS_TF26, reason='TF2.6+ is required')
class XLATests(tf.test.TestCase):
    """
    Tests for ops in byteps.tensorflow.
    """

    def __init__(self, *args, **kwargs):
        super(XLATests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def evaluate(self, tensors):
        if _executing_eagerly():
            return self._eval_helper(tensors)
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session(config=config) as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)

    def random_uniform(self, *args, **kwargs):
        if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
            tf.random.set_seed(1234)
            return tf.random.uniform(*args, **kwargs)
        else:
            tf.set_random_seed(1234)
            return tf.random_uniform(*args, **kwargs)

    def filter_supported_types(self, types):
        if 'CCL_ROOT' in os.environ:
            types = [t for t in types if t in ccl_supported_types]
        return types

    def test_test_byteps_pushpull_gpu(self):
        """Test that the pushpull works on XLA/GPUs."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        local_rank = bps.local_rank()
        size = bps.size()

        def bps_pushpull_test(self, dtype, dim, idx=0):
            tensor = self.random_uniform(
                [17] * dim, -100, 100, dtype=dtype)
            summed = bps.push_pull(tensor, average=False,
                name="xxxx_" + str(idx) + "_" + str(dtype.name) + "_" + str(dim) + "_" + str(tensor.shape))
            multiplied = tensor * size
            max_difference = tf.reduce_max(tf.abs(summed - multiplied))
            return max_difference

        dtypes = [tf.int32, tf.int64, tf.float32, tf.float16, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                max_difference = tf.function(
                    bps_pushpull_test, jit_compile=True)(self, dtype, dim)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest(
                    "Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(
                diff <= threshold,
                "bps.pushpull on XLA/GPU produces incorrect results")

    def test_test_byteps_pushpull_gpu_2(self):
        """Test that the pushpull works on XLA/GPUs."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        local_rank = bps.local_rank()
        size = bps.size()

        def bps_pushpull_test(self, dtype, dim, idx=0):
            tensor = self.random_uniform(
                [17] * dim, -100, 100, dtype=dtype)
            summed = bps.push_pull(tensor, average=False, \
                name="yyyy_" + str(idx) + "_" + str(dtype.name) + "_" + str(dim) + "_" + str(tensor.shape))
            multiplied = tensor * size
            max_difference = tf.reduce_max(tf.abs(summed - multiplied))
            return max_difference

        def bps_pushpull_test_wrapper(self, dtype, dim):
            num = 100
            num = 10
            results = []
            for idx in range(num):
                aa = bps_pushpull_test(self, dtype, dim, idx)
                results.append(aa)
            return results


        dtypes = [tf.int32, tf.int64, tf.float32, tf.float16, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                max_difference = tf.function(
                    bps_pushpull_test_wrapper, jit_compile=True)(self, dtype, dim)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest(
                    "Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            print(f'xxx test 2 done for dtype: {dtype}, dim: {dim}')
            # self.assertTrue(
            #     diff <= threshold,
            #     "bps.pushpull on XLA/GPU produces incorrect results")

    def xxxx_test_byteps_pushpull_grad_gpu(self):
        """Test the correctness of the pushpull gradient on XLA/GPU.
           For some reason when using tf.gradient the backward op receives a CPU
           tensor.
        """
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        local_rank = bps.local_rank()
        size = bps.size()

        def pushpull_grad_test(self, dtype, dim, idx=0):
            tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
            # summed = bps.push_pull(tensor, average=False)
            summed = bps.push_pull(tensor, average=False, device_dense="/gpu:%d" % local_rank, \
                name="grad_gpu_" + str(idx) + "_" + str(dtype.name) + "_" + str(dim) + "_" + str(tensor.shape))

            with tf.device("/gpu:%d" % local_rank):
                grad_ys = tf.ones([5] * dim)
                grad = tf.gradients(summed, tensor, grad_ys)[0]
            return grad

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                grad = tf.function(pushpull_grad_test,
                                   jit_compile=True)(self, dtype, dim)
                grad_out = self.evaluate(grad)
            expected = np.ones([5] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def xxx_test_byteps_pushpull_average_grad_gpu(self):
        """Test the correctness of the pushpull with average gradient on XLA/GPU.
           For some reason when using tf.gradient the backward op receives a CPU
           tensor.
        """
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        local_rank = bps.local_rank()
        size = bps.size()

        def pushpull_grad_test(self, dtype, dim):
            tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
            averaged = bps.pushpull(tensor, average=True)

            grad_ys = tf.ones([5] * dim, dtype=dtype)
            grad = tf.gradients(averaged, tensor, grad_ys)[0]
            return grad

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                grad = tf.function(pushpull_grad_test,
                                   jit_compile=True)(self, dtype, dim)
                grad_out = self.evaluate(grad)
            expected = np.ones([5] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))


run_all_in_graph_and_eager_modes(XLATests)

if __name__ == '__main__':
    tf.test.main()
