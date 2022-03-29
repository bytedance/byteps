import tensorflow as tf
import numpy as np
import os

import argparse
parser = argparse.ArgumentParser(description='Tensorflow tests')
parser.add_argument('--backend', type=str, default='byteps')
parser.add_argument('--iter', type=int, default=250)

is_use_pull = int(os.environ.get('BYTEPS_ALL2ALL_USE_PULL', 0))
test_cpu_only = int(os.environ.get('TEST_CPU_ONLY', 0))
test_sanity = int(os.environ.get('TEST_SANITY', 0))
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
        tf.config.experimental.set_visible_devices(gpus[bps.local_rank()], 'GPU')
    except RuntimeError as e:
        print(e)
        exit()

class TensorFlowTests:
    """
    Tests for ops in byteps.tensorflow.
    """
    def __init__(self):
        self.rank = bps.rank()
        self.size = bps.size()
        print(bps.__file__)

    def mse(self, std_value, cmp_value):
        abs_dis = np.abs(cmp_value - std_value)
        abs_std = np.abs(std_value) + 0.00001
        mse = abs_dis / abs_std
        return np.max(mse)

    def broadcast_variables(self, total_niter=args.iter):
        rank = self.rank
        size = self.size
        dtypes = [tf.int64, tf.float32, tf.int32]
        num_vars = 100
        dim_1 = 2
        idx_dtype = tf.int32
        rng = np.random.default_rng(size)
        tf.compat.v1.random.set_random_seed(size)
        total_niter = 1
        for dtype in dtypes:
            niter = 0
            while niter < total_niter:
                print(f'rank={rank}/{size}, dtype={dtype.name}, niter={niter} starting', flush=True)
                dim_0 = rng.integers(low=10, high=100, size=num_vars)
                with tf.device("/gpu:0"):
                    my_tensors = [tf.random.uniform(shape=[dim_0[i], dim_1],
                                  maxval=1000, dtype=dtype) for i in range(0, num_vars)]
                my_vars = []
                for i, t in enumerate(my_tensors):
                    name="my_var_eager_" + dtype.name + "_" + str(niter) + "_" + str(i)
                    my_vars.append(tf.Variable(t, name=name))
                ground_truth = [tf.identity(item) for item in my_vars]
                if rank != 0:
                    [tf.compat.v1.assign_sub(item, item) for item in my_vars]
                bps.broadcast_variables(my_vars, 0)

                succeded = 0
                failed = 0
                for i in range(len(my_vars)):
                    cur_mse = self.mse(ground_truth[i], my_vars[i])
                    if cur_mse > 0.01:
                        failed += 1
                        print(f"Failed to broadcast variable to rank {bps.rank()}: "
                              f"expecting: {ground_truth[i].numpy()} got: "
                              f"{my_vars[i].numpy()} with mse {cur_mse}.")
                    else:
                        succeded += 1
                print(f"iter {niter}/{total_niter} Broadcast global variables to rank: "
                      f"{bps.rank()}: {succeded}  successful, {failed} failed")
                assert failed == 0
                succeded = 0
                failed = 0
                niter += 1

        print(f'===== end test broadcast_variables =====', flush=True)

    def broadcast_variables_hook(self, total_niter=args.iter):
        tf.compat.v1.disable_eager_execution()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(bps.local_rank())

        rank = self.rank
        size = self.size
        dtypes = [tf.int64, tf.float32, tf.int32]
        num_vars = 100
        dim_1 = 2
        idx_dtype = tf.int32
        rng = np.random.default_rng(size)
        tf.compat.v1.random.set_random_seed(size)
        total_niter = 1
        for dtype in dtypes:
            niter = 0
            while niter < total_niter:
                print(f'rank={rank}/{size}, dtype={dtype.name}, niter={niter} starting', flush=True)
                tf.compat.v1.reset_default_graph()
                dim_0 = rng.integers(low=10, high=100, size=num_vars)
                with tf.device("/gpu:0"):
                    my_tensors = [tf.ones(shape=[dim_0[i], dim_1], dtype=dtype) for i in range(0, num_vars)]

                my_vars = []
                my_truth_vars = []
                for i, t in enumerate(my_tensors):
                    name="my_var_" + dtype.name + "_" + str(niter) + "_" + str(i)
                    my_vars.append(tf.Variable(t * (rank + 1), name=name))
                    name="my_truth_var_" + dtype.name + "_" + str(niter) + "_" + str(i)
                    my_truth_vars.append(tf.Variable(t, name=name))

                init_op = tf.compat.v1.global_variables_initializer()
                sess = tf.compat.v1.Session(config=config)
                sess.run(init_op)
                ground_truth = sess.run(my_truth_vars)
                hooks = [
                    bps.BroadcastGlobalVariablesHook(0),
                ]
                with tf.compat.v1.train.MonitoredTrainingSession(hooks=hooks,
                                           config=config) as mon_sess:
                    received_vars = mon_sess.run(my_vars)

                succeded = 0
                failed = 0
                for i in range(len(my_vars)):
                    cur_mse = self.mse(ground_truth[i], received_vars[i])
                    if cur_mse > 0.01:
                        failed += 1
                        print(f"Failed to broadcast variable {i} to rank {bps.rank()}: "
                              f"expecting:\n{ground_truth[i]} got:\n"
                              f"{received_vars[i]} with mse {cur_mse}.")
                    else:
                        succeded += 1
                print(f"iter {niter}/{total_niter} Broadcast global variables to rank: "
                      f"{bps.rank()}: {succeded}  successful, {failed} failed")
                assert failed == 0
                succeded = 0
                failed = 0
                niter += 1

        print(f'===== end test broadcast_variables hook =====', flush=True)

tests = TensorFlowTests()
tests.broadcast_variables()
tests.broadcast_variables_hook()
