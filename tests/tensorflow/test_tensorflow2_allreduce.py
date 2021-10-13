import tensorflow as tf
import numpy as np
import os
import time
import sys
import itertools
from byteps.tensorflow.util import _executing_eagerly

import argparse
parser = argparse.ArgumentParser(description='Tensorflow tests')
parser.add_argument('--backend', type=str, default='byteps')
parser.add_argument('--iter', type=int, default=250)
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()
if args.backend == 'byteps':
    import byteps.tensorflow as bps
    bps.init(lazy=False)
else:
    import horovod.tensorflow as bps
    bps.init()

print(f'loading byteps from {bps.__file__}')

args.iter = int(os.environ.get('TEST_NUM_ITER', args.iter))
args.device = os.environ.get('TEST_ALLREDUCE_DEVICE', args.device)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        exit()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(0)
class TensorFlowTests(tf.test.TestCase):
    """
    Tests for ops in byteps.tensorflow.
    """
    def __init__(self):
        self.rank = bps.rank()
        self.size = bps.size()
        print(bps.__file__)

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

    def test_byteps_allreduce_sum_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        rank = self.rank
        size = self.size
        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        devices = ["/cpu:0"]
        for dtype, dim, device in itertools.product(dtypes, dims, devices):
            with tf.device(device):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = bps.push_pull(tensor, average=False,
                        name=f'allreduce_{dtype.name}_{dim}_{device.strip("/")}')
                multiplied = tensor * size
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("BytePS cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            assert diff <= threshold, "bps.push_pull produced incorrect results"
            print(f"bps.push_pull test success!")
            print(f'name: allreduce_{dtype.name}_{dim}_{device.strip("/")}')

    def test_byteps_allreduce_average_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        rank = self.rank
        size = self.size
        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                averaged = bps.push_pull(tensor, average=True,
                        name=f'allreduce_{dtype.name}_{dim}_cpu_0')
            max_difference = tf.reduce_max(tf.abs(tf.cast(averaged, dtype=dtype) - tensor))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                print("BytePS cluster too large for precise multiplication comparison", flush=True)

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "bps.push_pull produced incorrect results")

    def test_byteps_allreduce_sum_gpu(self):
        """Test on GPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        rank = self.rank
        size = self.size
        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        devices = ["/gpu:0"]
        for dtype, dim, device in itertools.product(dtypes, dims, devices):
            with tf.device(device):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = bps.push_pull(tensor, average=False,
                        name=f'allreduce_{dtype.name}_{dim}_{device.strip("/")}')
                multiplied = tensor * size
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("BytePS cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            assert diff <= threshold, "bps.push_pull produced incorrect results"
            print(f"bps.push_pull test success!")
            print(f'name: allreduce_{dtype.name}_{dim}_{device.strip("/")}')

    def test_byteps_allreduce_sum_gpu_throughput(self, niters=args.iter):
        """Speed tests of GPU allreduce."""
        rank = self.rank
        size = self.size
        dtypes = [tf.float32, tf.float16]
        dims = [3]
        devices = ["/gpu:0"]
        interval = 10
        for dtype, dim, device in itertools.product(dtypes, dims, devices):
            t0 = time.time()
            for i in range(niters):
                with tf.device(device):
                    tensor = self.random_uniform(
                        [128] * dim, -100, 100, dtype=dtype)
                    summed = bps.push_pull(tensor, average=False,
                        name=f'allreduce_{dtype.name}_{dim}_{device.strip("/")}')
                if i % interval == 0:
                    t1 = time.time()
                    latency = (t1 - t0) / interval
                    len = int(tf.size(tensor).numpy()) * (2 if dtype == tf.float16 else 4)
                    goodput = len * 8 / latency / 1e9
                    if bps.local_rank() == 0:
                        print(f'iter {i} \t goodput {goodput:.4} Gb/s \t latency {(latency * 1e3):.4} ms \t size {len} bytes')
                        sys.stdout.flush()
                    t0 = time.time()
            print(f"bps.push_pull test success!")


tests = TensorFlowTests()
if args.device == 'cpu':
    tests.test_byteps_allreduce_sum_cpu()
    tests.test_byteps_allreduce_average_cpu()
else:
    tests.test_byteps_allreduce_sum_gpu()
