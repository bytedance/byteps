import tensorflow as tf
import numpy as np
import time
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

    def test_telemtry(self):
        telemetries = bps.get_telemetry()
        for entry in telemetries:
            name, mean, stdev, count = entry
            assert 'telemetry' not in name
        self.test_all2all(total_niter=2, src_device='cpu', dst_device='cpu', prefix="telemetry_")

        def check_telemetry(entries, max_size):
            entries = filter(lambda x: 'telemetry' in str(x[0]), entries)
            assert len(list(entries)) > 0
            if max_size:
                assert len(list(entries)) <= max_size
            for entry in entries:
                name, mean, stdev, count = entry
                assert name
                assert mean > 0
                assert stdev > 0
                assert count > 1
        telemetries = bps.get_telemetry()
        check_telemetry(telemetries, None)
        # check with user-provided size
        telemetries = bps.get_telemetry(size=10000)
        check_telemetry(telemetries, 10000)

    def test_all2all_autograd(self, total_niter=args.iter, src_gpu=False, dst_gpu=False):
        dtype = tf.float32
        rank = self.rank
        size = self.size
        vector_dim = 2
        # every worker should have the same size
        rng = np.random.default_rng(54321)
        niter = 0
        if src_gpu and dst_gpu:
            alltoall_fn = bps.alltoall
            name = 'autograd_gpu2gpu'
        elif not src_gpu and dst_gpu:
            alltoall_fn = bps.alltoall_cpu2gpu
            name = 'autograd_cpu2gpu'
        elif not src_gpu and not dst_gpu:
            alltoall_fn = bps.alltoall
            name = 'autograd_cpu2cpu'
        else:
            alltoall_fn = bps.alltoall_gpu2cpu
            name = 'autograd_gpu2cpu'

        print(f'start alltoall autograd tests, src_gpu={src_gpu}, dst_gpu={dst_gpu}', flush=True)
        while niter < total_niter:
            p2p_matrix = rng.integers(low=0, high=10, size=size * size).reshape(size, size) 
            splits_list = list(p2p_matrix[rank])
            recv_splits_list = list(p2p_matrix[:, rank])
            splits = tf.constant(splits_list, dtype=tf.int32)
            recv_splits = tf.constant(recv_splits_list, dtype=tf.int32)
            with tf.device("/gpu:0" if src_gpu else "/cpu:0"):
                tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                w = tf.Variable(tensor)
                with tf.GradientTape() as tape:
                    loss = alltoall_fn(w, splits=splits, recv_splits=recv_splits, name=f'{name}_iter_{niter % 5}')
                grad = tape.gradient(loss, w)
                assert grad.shape == tensor.shape, "the shapes of tensor and grad are not identical."
                print(f'DONE iter={niter}, loss={loss.shape}, device={loss.device}')
                niter += 1

    def test_all2all_invalid_splits(self):
        """Test alltoall with invalid splits/recv_splits."""
        print(f'test all2all invalid splits. You may see OP_REQUIRES errors from TF', flush=True)
        rank = self.rank
        size = self.size
        dtype = tf.float32
        vector_dim = 2
        # every worker should have the same size
        rng = np.random.default_rng(size)
        alltoall_fn = bps.alltoall
        p2p_matrix = rng.integers(low=0, high=10, size=size*size).reshape(size,size)
        splits_list = list(p2p_matrix[rank])
        recv_splits_list = list(p2p_matrix[:, rank])
        def check_invalid_alltoall(split_delta, recv_split_delta, splits_list, recv_splits_list):
            splits_list_copy = splits_list.copy()
            recv_splits_list_copy = recv_splits_list.copy()
            splits_list_copy[0] += split_delta
            recv_splits_list_copy[0] += recv_split_delta
            with tf.device("/cpu:0"):
                splits = tf.constant(splits_list_copy, dtype=tf.int32)
                recv_splits = tf.constant(recv_splits_list_copy, dtype=tf.int32)
                tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype)
                try:
                    alltoall_fn(tensor, splits=splits, recv_splits=recv_splits, name=f'test_invalid_splits')
                    assert False
                except tf.errors.InvalidArgumentError:
                    pass
        # check split with negative values
        check_invalid_alltoall(-100, 0, splits_list, recv_splits_list)
        # check recv_split with negative values
        check_invalid_alltoall(0, -100, splits_list, recv_splits_list)
        # check split inconsistent with shape[0]
        check_invalid_alltoall(1, 0, splits_list, recv_splits_list)
        print(f'DONE testing all2all invalid splits', flush=True)

    def test_all2all(self, total_niter=args.iter, compression=bps.Compression.none,
                     src_device='cpu', dst_device='cpu', prefix=""):
        """Test that alltoall correctly send/recv tensors with given recv_splits."""
        print(f'test all2all {src_device}->{dst_device}', flush=True)
        rank = self.rank
        size = self.size
        # TODO: record type info in declare_tensor
        dtypes = [tf.int64, tf.float32, tf.int32]
        vector_dim = 2
        idx_dtype = tf.int32
        # every worker should have the same size
        rng = np.random.default_rng(size)
        alltoall_fn = bps.alltoall
        if src_device == 'cpu' and dst_device == 'gpu':
            alltoall_fn = bps.alltoall_cpu2gpu
        if src_device == 'gpu' and dst_device == 'cpu':
            alltoall_fn = bps.alltoall_gpu2cpu
        for dtype in dtypes:
            niter = 0
            while niter < total_niter:
                p2p_matrix = rng.integers(low=0, high=10, size=size*size).reshape(size,size)
                splits_list = list(p2p_matrix[rank])
                recv_splits_list = list(p2p_matrix[:, rank])
                print(f'rank={rank}/{size}, split={splits_list}, recv_split={recv_splits_list},\
                        mat={p2p_matrix.tolist()}, dim={vector_dim}, {dtype}', flush=True)
                with tf.device(f"/cpu:0"):
                    splits = tf.constant(splits_list, dtype=tf.int32)
                    recv_splits = tf.constant(recv_splits_list, dtype=tf.int32)
                with tf.device(f"/{src_device}:0"):
                    tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                with tf.device(f"/{dst_device}:0"):
                    name = f'{prefix}test_{src_device}_{dst_device}_iter_{niter % 5}'
                    result = alltoall_fn(tensor, splits=splits, recv_splits=recv_splits,
                                         name=name, compression=compression)
                    print(f'DONE iter={niter}, shape={result.shape}, {result.device}')
                    index = 0
                    for i in range(size):
                        begin = index
                        end = index + recv_splits_list[i]
                        subset = result[begin:end]
                        val = i + 1
                        assert np.sum(subset != val) == 0, (subset, val, result)
                        index = end
                    niter += 1

    def test_all2all_no_recv_splits(self, total_niter=args.iter, compression=bps.Compression.none):
        """Test on CPU that the alltoall correctly send/recv tensors without recv_splits."""
        print('test all2all_no_recv_splits', flush=True)
        dtype = tf.float32
        rank = self.rank
        size = self.size
        vector_dim = 2
        rng = np.random.default_rng(12345)
        niter = 0
        while niter < total_niter:
            p2p_matrix = rng.integers(low=0, high=10, size=size*size).reshape(size,size)
            splits_list = list(p2p_matrix[rank])
            recv_splits_list = list(p2p_matrix[:, rank])
            print(f'rank={rank}, size={size}, split={splits_list}, recv_split={recv_splits_list},\
                    matrix={p2p_matrix.tolist()}', flush=True)
            with tf.device("/cpu:0"):
                splits = tf.constant(splits_list, dtype=tf.int32)
                tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                result, recv_split = bps.alltoall(tensor, splits=splits, name=f'no_recv_iter_{niter % 5}',
                                                  with_size=True, compression=compression)
                print(f'DONE iter={niter}, shape={result.shape}', flush=True)
                index = 0
                for i in range(2):
                    begin = index
                    end = index + recv_splits_list[i]
                    subset = result[begin:end]
                    val = i + 1
                    assert np.sum(subset != val) == 0, (subset, val, result)
                    index = end
                recv_split = recv_split.numpy()
                assert np.sum(recv_split != p2p_matrix[:, rank]) == 0, (recv_splits_list, recv_split)
                niter += 1

    def test_self_send_recv(self):
        # same worker
        num_elements = 10
        dtype, np_dtype = tf.float32, 'float32'
        def receiver(from_rank):
            device = tf.device('GPU:' + str(self.rank))
            with device:
                tensor = tf.zeros(num_elements, dtype=dtype)
                bps.recv_async(tensor, from_rank, name='test')
                expected = np.arange(0, num_elements, step=1.0, dtype=np_dtype)
                assert np.sum(tensor != expected) == 0, tensor

        def sender(to_rank):
            device = tf.device('GPU:' + str(self.rank))
            with device:
                tensor = tf.range(0, num_elements, 1.0, dtype=dtype)
                bps.send_async(tensor, to_rank, name='test')

        sender(to_rank=self.rank)
        receiver(from_rank=self.rank)
        print('test_self_send_recv DONE', flush=True)

    def test_send_recv(self):
        # cross worker
        num_elements = 10
        dtype, np_dtype = tf.float32, 'float32'
        def receiver(from_rank):
            device = tf.device('GPU:' + str(self.rank))
            with device:
                tensor = tf.zeros(num_elements, dtype=dtype)
                bps.recv_async(tensor, from_rank, name='test')
                expected = np.arange(0, num_elements, step=1.0, dtype=np_dtype)
                assert np.sum(tensor != expected) == 0, tensor

        def sender(to_rank):
            device = tf.device('GPU:' + str(self.rank))
            with device:
                tensor = tf.range(0, num_elements, 1.0, dtype=dtype)
                bps.send_async(tensor, to_rank, name='test')

        if self.rank == 0:
            sender(to_rank=1)
        elif self.rank == 1:
            receiver(from_rank=0)
        else:
            raise RuntimeError
        print('test_send_recv DONE', flush=True)

    def validate_a2a(self, recv_splits_list, result, size, rank):
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
        print(f'rank {rank} result is correct')

    # TODO: test other dtypes
    def test_all2all_benchmark(self, total_niter=args.iter, dst_gpu=False, src_gpu=False):
        """Test on CPU that the alltoall correctly send/recv tensors with given recv_splits."""
        dtype, int_dtype = tf.float32, tf.int32
        rank = self.rank
        size = self.size
        vector_dim = 1
        niter = 0
        total_len = int(os.environ.get('TOTAL_LEN', '2048000'))
        len_per_worker = int(total_len / size)
        assert total_len % size == 0
        p2p_matrix = np.array([len_per_worker]*(size*size)).reshape(size, size)
        splits_list = list(p2p_matrix[rank])
        recv_splits_list = list(p2p_matrix[:, rank])
        print(f'rank={rank}, size={size}, split={splits_list}, recv_split={recv_splits_list}, matrix={p2p_matrix.tolist()}', flush=True)
        splits = tf.constant(splits_list, dtype=int_dtype)
        recv_splits = tf.constant(recv_splits_list, dtype=int_dtype)
        with tf.device("/gpu:0" if src_gpu else "/cpu:0"):
            tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
        t0 = time.time()
        interval = 10
        if dst_gpu:
            if src_gpu:
                alltoall_fn = bps.alltoall
            else:
                alltoall_fn = bps.alltoall_cpu2gpu
        else:
            if src_gpu:
                alltoall_fn = bps.alltoall_gpu2cpu
            else:
                alltoall_fn = bps.alltoall
        while niter < total_niter:
            if args.backend == 'byteps':
                name = 'data_'
                if src_gpu:
                    if dst_gpu:
                        name += 'g2g'
                    else:
                        name += 'g2c'
                else:
                    if dst_gpu:
                        name += 'c2g'
                    else:
                        name += 'c2c'
                result = alltoall_fn(tensor, splits=splits, recv_splits=recv_splits, name=name)
            else:
                result = bps.alltoall(tensor, splits=splits)
            niter += 1
            if niter % interval == 0:
                t1 = time.time()
                latency = (t1-t0)/interval*1000
                goodput = total_len*32*interval/(t1-t0)/1000000000
                print(f'DONE iter={niter}, latency={latency:.3} ms, Goodput={goodput:.4} Gb/s', flush=True)
                t0 = time.time()
        print(f'Finish all2all_benchmark, srcdev={tensor.device}, dstdev={result.device}')
        self.validate_a2a(recv_splits_list, result, size, rank)

    def test_all2all_group(self, total_niter=args.iter, compression=bps.Compression.none,
                     src_device='gpu', dst_device='gpu'):
        """Test on CPU that the alltoall correctly send/recv tensors with given recv_splits."""
        print(f'test all2all group {src_device}->{dst_device}', flush=True)
        rank = self.rank
        size = self.size
        # TODO: record type info in declare_tensor
        dtypes = [tf.int64, tf.float32, tf.int32]
        vector_dim = 1
        idx_dtype = tf.int32
        # every worker should have the same size
        rng = np.random.default_rng(size)
        alltoall_fn = bps.alltoall
        if src_device == 'cpu' and dst_device == 'gpu':
            alltoall_fn = bps.alltoall_cpu2gpu
        if src_device == 'gpu' and dst_device == 'cpu':
            alltoall_fn = bps.alltoall_gpu2cpu
        for dtype in dtypes:
            niter = 0
            while niter < total_niter:
                p2p_matrix = rng.integers(low=1, high=10, size=size*size).reshape(size,size)
                splits_list = list(p2p_matrix[rank])
                recv_splits_list = list(p2p_matrix[:, rank])
                print(f'rank={rank}/{size}, split={splits_list}, recv_split={recv_splits_list}, {dtype}', flush=True)
                with tf.device(f"/cpu:0"):
                    splits = tf.constant(splits_list, dtype=tf.int32)
                    recv_splits = tf.constant(recv_splits_list, dtype=tf.int32)
                with tf.device(f"/{src_device}:0"):
                    tensors = [tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                               for i in range(len(splits_list))]
                with tf.device(f"/{dst_device}:0"):
                    results = alltoall_fn(tensors, splits=splits, recv_splits=recv_splits,
                                         name=f'alltoall_group_test_{src_device}_{dst_device}_iter_{niter % 5}',
                                         compression=compression)
                    print(f'AlltoAll group tests: Done iter={niter}, shape={results[0].shape}')
                    # validate result
                    index = 0
                    for i in range(size):
                        begin = index
                        end = index + recv_splits_list[i]
                        subset = results[i]
                        val = i + 1
                        assert np.sum(subset != val) == 0, (subset, val, results[i])
                        index = end
                    niter += 1

    def test_all2all_group_benchmark(self, total_niter=args.iter, dst_gpu=False, src_gpu=False):
        """Test on CPU that the alltoall correctly send/recv tensors with given recv_splits."""
        dtype, int_dtype = tf.float32, tf.int32
        if args.backend == 'byteps':
            int_dtype = tf.int32
        else:
            assert False, 'horovod is not supported for this test'
        rank = self.rank
        size = self.size
        vector_dim = 1
        niter = 0
        total_len = int(os.environ.get('TOTAL_LEN', '2048000'))
        len_per_worker = int(total_len / size)
        assert total_len % size == 0
        p2p_matrix = np.array([len_per_worker]*(size*size)).reshape(size, size)
        splits_list = list(p2p_matrix[rank])
        recv_splits_list = list(p2p_matrix[:, rank])
        print(f'rank={rank}, size={size}, split={splits_list}, recv_split={recv_splits_list}, matrix={p2p_matrix.tolist()}', flush=True)
        splits = tf.constant(splits_list, dtype=int_dtype)
        recv_splits = tf.constant(recv_splits_list, dtype=int_dtype)
        with tf.device("/gpu:0" if src_gpu else "/cpu:0"):
            tensors = [tf.ones([splits_list[i], vector_dim], dtype=dtype) * (rank + 1) 
                       for i in range(len(splits_list))]
        print('start group all2all test')
        t0 = time.time()
        interval = 10
        name = 'group_test_data_'
        if dst_gpu:
            if src_gpu:
                name += 'g2g'
                alltoall_fn = bps.alltoall
            else:
                name += 'c2g'
                alltoall_fn = bps.alltoall_cpu2gpu
        else:
            if src_gpu:
                name += 'g2c'
                alltoall_fn = bps.alltoall_gpu2cpu
            else:
                name += 'c2c'
                alltoall_fn = bps.alltoall
        while niter < total_niter:
            results = alltoall_fn(tensors, splits=splits, recv_splits=recv_splits, name=name)
            niter += 1
            if niter % interval == 0:
                t1 = time.time()
                latency = (t1-t0)/interval*1000
                goodput = total_len*32*interval/(t1-t0)/1000000000
                print(f'DONE iter={niter}, latency={latency:.3} ms, Goodput={goodput:.4} Gb/s', flush=True)
                t0 = time.time()
        print(f'Finish all2all_group_benchmark, srcdev={tensors[0].device}, dstdev={results[0].device}')
        self.validate_a2a(recv_splits_list, results, size, rank)

    def test_all2all_group_autograd(self, total_niter=args.iter, src_gpu=False, dst_gpu=False):
        dtype = tf.float32
        rank = self.rank
        size = self.size
        vector_dim = 1
        # every worker should have the same size
        rng = np.random.default_rng(54321)
        niter = 0
        if src_gpu and dst_gpu:
            alltoall_fn = bps.alltoall
            name = 'autograd_group_gpu2gpu'
        elif not src_gpu and dst_gpu:
            alltoall_fn = bps.alltoall_cpu2gpu
            name = 'autograd_group_cpu2gpu'
        elif not src_gpu and not dst_gpu:
            alltoall_fn = bps.alltoall
            name = 'autograd_group_cpu2cpu'
        else:
            alltoall_fn = bps.alltoall_gpu2cpu
            name = 'autograd_group_gpu2cpu'
        print(f'start alltoall autograd group tests, src_gpu={src_gpu}, dst_gpu={dst_gpu}', flush=True)
        while niter < total_niter:
            # need to use fixed length for each rank, 
            # since tf op requires identical shapes for all inputs during BP
            p2p_matrix = np.array([1]*(size*size)).reshape(size, size)
            splits_list = list(p2p_matrix[rank])
            recv_splits_list = list(p2p_matrix[:, rank])
            splits = tf.constant(splits_list, dtype=tf.int32)
            recv_splits = tf.constant(recv_splits_list, dtype=tf.int32)
            with tf.device("/gpu:0" if src_gpu else "/cpu:0"):
                tensors = [tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                          for _ in range(len(splits_list))]
                group_w = [tf.Variable(tensor) for tensor in tensors] 
                with tf.GradientTape() as tape:
                    group_loss = alltoall_fn(group_w, splits=splits, recv_splits=recv_splits, name=f'{name}_autograd_group_iter_{niter % 5}')
                grad = tape.gradient(group_loss, group_w)
                print(f'DONE iter={niter}, loss={group_loss[0].shape}, device={group_loss[0].device}')
                niter += 1

    def test_allreduce(self):
        print('test_allreduce', flush=True)
        total_niter = 10
        dtype = tf.float32
        rank = bps.rank()
        size = bps.size()
        niter = 0
        total_len = 0
        assert total_len % size == 0
        FIRST_DENSE_DIM = 1024
        print(f'rank={rank}, size={size}', flush=True)
        with tf.device("/cpu:0"):
            tensor_size = FIRST_DENSE_DIM
            total_len = tensor_size
            print(f'rank={rank}, tenosr_size={tensor_size}', flush=True)
            tensor = tf.ones([tensor_size, 1], dtype=dtype) * (rank + 1)
            while niter < total_niter:
                # forward
                result = bps.push_pull(tensor, name=f'test_allreduce_iter_{niter % 5}')
                expected = (1 + size + 1) * size / 2.0
                assert np.sum(result != expected), (result, expected)
                niter += 1
                print(f'DONE iter={niter}', flush=True)

tests = TensorFlowTests()


# FIXME: send to myself hangs
# tests.test_self_send_recv()
# tests.test_send_recv()

# TODO: remove this when we fix direct response
is_direct_resp = int(os.environ.get('BYTEPS_SERVER_DIRECT_RESPONSE', 0))
tests.test_telemtry()
tests.test_allreduce()
tests.test_all2all_invalid_splits()
tests.test_all2all(src_device='cpu', dst_device='cpu')
if test_sanity:
    exit()


if is_direct_resp > 0:
    tests.test_all2all(compression=bps.Compression.fp16)
    if not is_use_pull:
        tests.test_all2all_no_recv_splits()
        tests.test_all2all_no_recv_splits(compression=bps.Compression.fp16)

if is_direct_resp == 0:
    tests.test_all2all_autograd(src_gpu=False, dst_gpu=False)
    tests.test_all2all_group_autograd(src_gpu=False, dst_gpu=False)
    tests.test_all2all_benchmark()
    tests.test_all2all_group(src_device='cpu', dst_device='cpu')
    if not test_cpu_only:
        tests.test_all2all_autograd(src_gpu=False, dst_gpu=True)
        tests.test_all2all_autograd(src_gpu=True, dst_gpu=False)
        tests.test_all2all_autograd(src_gpu=True, dst_gpu=True)
        tests.test_all2all_group_autograd(src_gpu=True, dst_gpu=True)
        tests.test_all2all_group_autograd(src_gpu=True, dst_gpu=False)
        tests.test_all2all_group_autograd(src_gpu=False, dst_gpu=True)
        tests.test_all2all(src_device='gpu', dst_device='gpu')
        tests.test_all2all(src_device='cpu', dst_device='gpu')
        tests.test_all2all(src_device='gpu', dst_device='cpu')
        tests.test_all2all_benchmark(dst_gpu=True, src_gpu=False)
        tests.test_all2all_benchmark(dst_gpu=False, src_gpu=True)
        tests.test_all2all_benchmark(dst_gpu=True, src_gpu=True)
        tests.test_all2all_group(src_device='gpu', dst_device='cpu')
        tests.test_all2all_group(src_device='cpu', dst_device='gpu')
        tests.test_all2all_group(src_device='gpu', dst_device='gpu')
        tests.test_all2all_group_benchmark(dst_gpu=True, src_gpu=True)
    if not is_use_pull:
        tests.test_all2all_no_recv_splits()

time.sleep(1)
