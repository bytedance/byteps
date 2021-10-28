import tensorflow as tf
import numpy as np
import time
import os

import argparse
parser = argparse.ArgumentParser(description='Tensorflow tests')
parser.add_argument('--backend', type=str, default='byteps')
parser.add_argument('--iter', type=int, default=10)

test_cpu_only = int(os.environ.get('TEST_CPU_ONLY', 0))
args = parser.parse_args()
if args.backend == 'byteps':
    import byteps.tensorflow as bps
    bps.init(lazy=False)
else:
    import horovod.tensorflow as bps
    bps.init()

print(bps)

args.iter = int(os.environ.get('TEST_NUM_ITER', args.iter))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
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

    def validate_alltoall(self, recv_splits_list, result, size, rank):
        index = 0
        for i in range(size):
            begin = index
            end = index + recv_splits_list[i]
            if isinstance(result, list):
                subset = result[i]
            else:
                subset = result[begin:end]
            val = i + 1
            assert np.sum(subset != val) == 0, (subset, val, result)
            index = end
        print(f'rank {rank} alltoall result is correct')

    def validate_allreduce(self, result, size, rank):
        expected = (1 + size + 1) * size / 2.0
        assert np.sum(result != expected), (result, expected)
        print(f'rank {rank} allreduce result is correct')

    @tf.function
    def mix_allreduce_alltoall(self, alltoall_fn, alltoall_tensors, dst_device, allreduce_tensor, 
                                splits, recv_splits, alltoall_name, allreduce_name, compression):
        with tf.device(f"/{dst_device}:0"):
            alltoall_results = alltoall_fn(alltoall_tensors, splits=splits, recv_splits=recv_splits,
                                name=alltoall_name, compression=compression)
        with tf.device("/gpu:0"):
            allreduce_result = bps.push_pull(allreduce_tensor, name=allreduce_name)
        return alltoall_results, allreduce_result

    def test_mix_all2all_allreduce(self, total_niter=args.iter, compression=bps.Compression.none,
                                    alltoall_src_dev='gpu', alltoall_dst_dev='gpu'):
        src_device, dst_device = alltoall_src_dev, alltoall_dst_dev
        print(f'===== start all2all {src_device}->{dst_device} & gpu allreduce mix test =====', flush=True)
        rank = self.rank
        size = self.size
        # TODO: record type info in declare_tensor
        dtypes = [tf.float32]
        vector_dim = 1
        # every worker should have the same size
        rng = np.random.default_rng(size)
        alltoall_fn = bps.alltoall
        if src_device == 'cpu' and dst_device == 'gpu':
            alltoall_fn = bps.alltoall_cpu2gpu
        if src_device == 'gpu' and dst_device == 'cpu':
            alltoall_fn = bps.alltoall_gpu2cpu
        for dtype in dtypes:
            for niter in range(total_niter):
                p2p_matrix = rng.integers(low=1, high=10, size=size*size).reshape(size,size)
                splits_list = list(p2p_matrix[rank])
                recv_splits_list = list(p2p_matrix[:, rank])
                with tf.device(f"/cpu:0"):
                    splits = tf.constant(splits_list, dtype=tf.int32)
                    recv_splits = tf.constant(recv_splits_list, dtype=tf.int32)
                with tf.device(f"/{src_device}:0"):
                    alltoall_tensors = [tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                               for i in range(len(splits_list))]
                with tf.device("/gpu:0"):
                    allreduce_tensor = tf.ones([1024, 1], dtype=dtype) * (rank + 1)
                alltoall_name=f'alltoall_group_test_{src_device}_{dst_device}_iter_{niter}'
                alltoall_results, allreduce_result = self.mix_allreduce_alltoall(alltoall_fn, alltoall_tensors, dst_device, allreduce_tensor, 
                                                                            splits, recv_splits, alltoall_name, 
                                                                            f'test_allreduce_iter_{niter}', compression)
                self.validate_allreduce(allreduce_result, size, rank)
                self.validate_alltoall(recv_splits_list, alltoall_results, size, rank)
                print(f'Done iter={niter}')

tests = TensorFlowTests()
tests.test_mix_all2all_allreduce(alltoall_src_dev='gpu', alltoall_dst_dev='cpu')
tests.test_mix_all2all_allreduce(alltoall_src_dev='cpu', alltoall_dst_dev='gpu')
tests.test_mix_all2all_allreduce(alltoall_src_dev='gpu', alltoall_dst_dev='gpu')
