import tensorflow as tf
import numpy as np
import os
import itertools
from byteps.tensorflow.util import _executing_eagerly

import argparse
parser = argparse.ArgumentParser(description='Tensorflow tests')
parser.add_argument('--backend', type=str, default='byteps')
parser.add_argument('--iter', type=int, default=250)

args = parser.parse_args()
if args.backend == 'byteps':
    import byteps.tensorflow as bps
    bps.init(lazy=False)
else:
    import horovod.tensorflow as bps
    bps.init()

print(f'loading byteps from {bps.__file__}')

args.iter = int(os.environ.get('TEST_NUM_ITER', args.iter))

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

same_shape = True

class TensorFlowTests():
    """
    Tests for ops in byteps.tensorflow.
    """
    def __init__(self):
        self.rank = bps.rank()
        self.size = bps.size()
        print(bps.__file__)

    def random_uniform(self, *args, **kwargs):
        if "seed" in kwargs:
            seed = kwargs["seed"]
        else:
            seed = 1234

        if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
            tf.random.set_seed(seed)
            return tf.random.uniform(*args, **kwargs)
        else:
            tf.set_random_seed(seed)
            return tf.random_uniform(*args, **kwargs)

    def validate_allgather(self, gathered, shape, minval, maxval, dtype):
        golden = [self.random_uniform(shape if same_shape else [shape[0] + i] + shape[1:], 
            minval, maxval, seed=i, dtype=dtype) for i in range(self.size)]
        golden = tf.concat(golden, 0)
        max_difference = tf.reduce_max(tf.abs(gathered - golden))
        threshold = 0

        assert max_difference <= threshold, "bps.allgather produced incorrect results"
        print(f"bps.allgather test success!", flush=True)

    def test_byteps_allgather(self):
        """Test on GPU that the allgather correctly runs on 1D, 2D, 3D tensors."""
        rank = self.rank
        size = self.size
        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        devices = ["/gpu:0"]
        for iter in range(args.iter):
            for dtype, dim, device in itertools.product(dtypes, dims, devices):
                with tf.device(device):
                    shape = [10] * dim
                    minval = -100
                    maxval = 100
                    tensor = self.random_uniform(shape if same_shape else [shape[0] + self.rank] + shape[1:], 
                        minval, maxval, seed=self.rank, dtype=dtype)
                    gathered = bps.allgather(tensor, same_shape=same_shape,
                        name=f'allgather_{dtype.name}_{dim}_{device.strip("/").replace(":", "_")}')
                    self.validate_allgather(gathered, shape, minval, maxval, dtype=dtype)

    def validate_allreduce(self, reduced, shape, minval, maxval, dtype):
        golden = [self.random_uniform(shape, minval, maxval, seed=i, dtype=dtype) for i in range(self.size)]
        golden = tf.reduce_sum(golden, 0)
        max_difference = tf.reduce_max(tf.abs(reduced - golden))

        if self.size <= 3 or dtype in [tf.int32, tf.int64]:
            threshold = 0
        elif dtype in [tf.float16]:
            threshold = 1
        elif self.size < 10:
            threshold = 1e-4
        elif self.size < 15:
            threshold = 5e-4

        assert max_difference <= threshold, "bps.allreduce produced incorrect results"
        print(f"bps.allreduce test success!", flush=True)

    # @tf.function
    def mix_allgather_allreduce(self, allgather_input, allreduce_input, device, allgather_name, allreduce_name):
        with tf.device(device):
            gathered = bps.allgather(allgather_input, same_shape=same_shape, name=allgather_name)

        with tf.device(device):
            reduced = bps.push_pull(allreduce_input, average=False, name=allreduce_name)

        return gathered, reduced

    def test_byteps_mix_allgather_allreduce(self):
        """Test on GPU that the allgather/allreduce correctly run on 1D, 2D, 3D tensors."""
        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        device = "/gpu:0"
        for iter in range(args.iter):
            for dtype, dim in itertools.product(dtypes, dims):
                with tf.device(device):
                    shape = [10] * dim
                    minval = -100
                    maxval = 100
                    allgather_input = self.random_uniform(shape if same_shape else [shape[0] + self.rank] + shape[1:],
                        minval, maxval, seed=self.rank, dtype=dtype)
                    allreduce_input = self.random_uniform(shape, minval, maxval, seed=self.rank, dtype=dtype)

                    allgather_name = f'allgather_{dtype.name}_{dim}_{device.strip("/").replace(":", "_")}_iter_{iter}'
                    allreduce_name = f'allreduce_{dtype.name}_{dim}_{device.strip("/").replace(":", "_")}_iter_{iter}'
                    gathered, reduced = self.mix_allgather_allreduce(allgather_input, allreduce_input, device, 
                                            allgather_name, allreduce_name)
                    self.validate_allgather(gathered, shape, minval, maxval, dtype=dtype)
                    self.validate_allreduce(reduced, shape, minval, maxval, dtype=dtype)
    
    def validate_allgather_autograd(self, grad, shape, minval, maxval, dtype):
        golden = [self.random_uniform(shape if same_shape else [shape[0] + i] + shape[1:], 
            minval, maxval, seed=i, dtype=dtype) for i in range(self.size)]
        golden = tf.concat(golden, 0) * 2.0
        if same_shape:
            golden = golden[self.rank * shape[0] : (self.rank + 1) * shape[0]]
        else:
            offset = 0
            for i in range(self.rank):
                offset += shape[0] + i
            golden = golden[offset : offset + shape[0] + self.rank]
            
        max_difference = tf.reduce_max(tf.abs(grad - golden))
        threshold = 5e-4

        assert max_difference <= threshold, "bps.allgather autograd produced incorrect results"
        print(f"bps.allgather autograd test success!", flush=True)

    def test_allgather_autograd(self):
        """Test of allgather autograd."""
        device = "/gpu:0"
        dtype=tf.float32
        for iter in range(args.iter):
            with tf.device(device):
                shape = [10]
                minval = -100
                maxval = 100
                tensor = self.random_uniform(shape if same_shape else [shape[0] + self.rank] + shape[1:],
                    minval, maxval, seed=self.rank, dtype=dtype)
                x = tf.Variable(tensor)
                name = f'allgather_{device.strip("/").replace(":", "_")}_iter_{iter}'
                with tf.GradientTape() as tape:
                    y = bps.allgather(x, same_shape=same_shape, name=name)
                    loss = y * y
                grad = tape.gradient(loss, x)
                self.validate_allgather_autograd(grad, shape, minval, maxval, dtype=dtype)

tests = TensorFlowTests()
for i in range(2):
    print(f"Test allgaher, same_shape is ", same_shape, flush=True)
    tests.test_byteps_allgather()
    tests.test_byteps_mix_allgather_allreduce()
    tests.test_allgather_autograd()
    same_shape = False

# same_shape = False
# tests.test_byteps_allgather()
# # tests.test_byteps_mix_allgather_allreduce()
# # tests.test_allgather_autograd()
