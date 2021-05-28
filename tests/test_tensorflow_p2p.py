import tensorflow as tf
import numpy as np
import time
import os

import argparse
parser = argparse.ArgumentParser(description='Tensorflow tests')
parser.add_argument('--backend', type=str, default='byteps')
parser.add_argument('--iter', type=int, default=250)

# no usage for now, temporarily add for compat
parser.add_argument('--rank', type=int, default=-1) 

args = parser.parse_args()
if args.backend == 'byteps':
    import byteps.tensorflow as bps
    bps.init(lazy=False)
else:
    import horovod.tensorflow as bps
    bps.init()

print(bps)

args.iter = int(os.environ.get('NUM_ITER', args.iter))

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


    def test_all2all_grad(self, total_niter=1):
        # w = tf.Variable([[1.0]])
        # with tf.GradientTape() as tape:
        #     loss = w * w

        # grad = tape.gradient(loss, w)
        # print('grad is ', grad)
        # exit()

        dtype = tf.float32
        rank = self.rank
        size = self.size
        vector_dim = 2
        # every worker should have the same size
        rng = np.random.default_rng(size)
        niter = 0
        while niter < total_niter:
            # FIXME: the test is limited to 2 workers only
            p2p_matrix = rng.integers(low=0, high=10, size=4).reshape(2,2) + 1
            splits_list = list(p2p_matrix[rank])
            recv_splits_list = list(p2p_matrix[:, rank])
            print(f'rank={rank}, size={size}, split={splits_list}, recv_split={recv_splits_list}, matrix={p2p_matrix.tolist()}', flush=True)
            with tf.device("/cpu:0"):
                splits = tf.constant(splits_list, dtype=tf.int64)
                recv_splits = tf.constant(recv_splits_list, dtype=tf.int64)
                tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                w = tf.Variable(tensor.numpy().tolist())
                with tf.GradientTape() as tape:
                    loss = bps.alltoall(w, splits=splits, recv_splits=recv_splits)

                grad = tape.gradient(loss, w)

                print(f'DONE iter={niter}, grad={grad}, loss={loss.shape}')
                # index = 0
                # for i in range(2):
                #     begin = index
                #     end = index + recv_splits_list[i]
                #     subset = result[begin:end]
                #     val = i + 1
                #     assert np.sum(subset != val) == 0, (subset, val)
                #     index = end
                niter += 1


    def test_all2all(self, total_niter=100, compression=bps.Compression.none):
        """Test on CPU that the alltoall correctly send/recv tensors with given recv_splits."""
        print('test_all2all', flush=True)
        rank = self.rank
        size = self.size
        # TODO: record type info in declare_tensor
        dtypes = [tf.int64, tf.float32, tf.int32]
        vector_dim = 2
        idx_dtype = tf.int32
        # every worker should have the same size
        rng = np.random.default_rng(size)
        for dtype in dtypes:
            niter = 0
            while niter < total_niter:
                p2p_matrix = rng.integers(low=0, high=10, size=size * size).reshape(size,size)
                splits_list = list(p2p_matrix[rank])
                recv_splits_list = list(p2p_matrix[:, rank])
                print(f'rank={rank}, size={size}, split={splits_list}, recv_split={recv_splits_list}, matrix={p2p_matrix.tolist()}, dim={vector_dim}, dtype={dtype}', flush=True)
                with tf.device("/cpu:0"):
                    splits = tf.constant(splits_list, dtype=idx_dtype)
                    recv_splits = tf.constant(recv_splits_list, dtype=idx_dtype)
                    tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                    result = bps.alltoall(tensor, splits=splits, recv_splits=recv_splits, name=f'test_iter{niter % 10}',
                                          compression=compression)
                    print(f'DONE iter={niter}, shape={result.shape}, {result.shape}')
                    index = 0
                    for i in range(size):
                        begin = index
                        end = index + recv_splits_list[i]
                        subset = result[begin:end]
                        val = i + 1
                        assert np.sum(subset != val) == 0, (subset, val, result)
                        index = end
                    niter += 1

    def test_all2all_cpu2gpu(self, total_niter=100, compression=bps.Compression.none):
        """Test on CPU that the alltoall correctly send/recv tensors with given recv_splits."""
        print('test_all2all_cpu2gpu', flush=True)
        rank = self.rank
        size = self.size
        # TODO: record type info in declare_tensor
        dtypes = [tf.int64, tf.float32, tf.int32]
        vector_dim = 2
        idx_dtype = tf.int32
        # every worker should have the same size
        rng = np.random.default_rng(size)
        for dtype in dtypes:
            niter = 0
            while niter < total_niter:
                p2p_matrix = rng.integers(low=0, high=10, size=size * size).reshape(size,size)
                splits_list = list(p2p_matrix[rank])
                recv_splits_list = list(p2p_matrix[:, rank])
                print(f'rank={rank}/{size}, split={splits_list}, recv_split={recv_splits_list}, matrix={p2p_matrix.tolist()}, dim={vector_dim}, dtype={dtype}', flush=True)
                with tf.device("/cpu:0"):
                    splits = tf.constant(splits_list, dtype=idx_dtype)
                    recv_splits = tf.constant(recv_splits_list, dtype=idx_dtype)
                    tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                result = bps.alltoall_cpu2gpu(tensor, splits=splits, recv_splits=recv_splits, name=f'test_iter_c2g_{niter % 10}',
                                              compression=compression)
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

    def test_all2all_no_recv_splits(self, total_niter=1000, compression=bps.Compression.none):
        """Test on CPU that the alltoall correctly send/recv tensors without recv_splits."""
        print('test_all2all_no_recv_splits', flush=True)
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
            print(f'rank={rank}, size={size}, split={splits_list}, recv_split={recv_splits_list}, matrix={p2p_matrix.tolist()}', flush=True)
            with tf.device("/cpu:0"):
                splits = tf.constant(splits_list, dtype=tf.int32)
                tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                result, recv_size = bps.alltoall(tensor, splits=splits, name=f'no_recv_iter_{niter % 10}',
                                                 with_size=True, compression=compression)
                print(f'DONE iter={niter}, shape={result.shape}')
                index = 0
                for i in range(2):
                    begin = index
                    end = index + recv_splits_list[i]
                    subset = result[begin:end]
                    val = i + 1
                    assert np.sum(subset != val) == 0, (subset, val, result)
                    index = end
                recv_size = recv_size.numpy()
                assert np.sum(recv_size != p2p_matrix[:, rank]) == 0, (recv_splits_list, recv_size)
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
            subset = result[begin:end]
            val = i + 1
            assert np.sum(subset != val) == 0, (subset, val, result)
            index = end
        print(f'rank {rank} result is correct')

    # TODO: test other dtypes
    def test_all2all_benchmark(self, total_niter=args.iter, dst_gpu=False, src_gpu=False):
        """Test on CPU that the alltoall correctly send/recv tensors with given recv_splits."""
        dtype, int_dtype = tf.float32, tf.int32
        if args.backend == 'byteps':
            int_dtype = tf.int32
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

    def test_all2all_fused_graph(self, total_niter=20, num_slots=20, num_groups=1):
        """Test on CPU that the alltoall correctly send/recv tensors with given recv_splits."""
        rank = self.rank
        size = self.size
        dtypes = [tf.float32, tf.int32, tf.int64]
        dtypes = [tf.int32, tf.float32]
        vector_dim = 2
        idx_dtype = tf.int32
        # every worker should have the same size
        rng = np.random.default_rng(size)
        tf.compat.v1.disable_eager_execution()
        for dtype in dtypes:
            niter = 0
            while niter < total_niter:
                outputs = []
                recv_splits_lists = []
                p2p_matrices = []
                print(f'iter={niter}', flush=True)
                for slot in range(num_slots):
                    p2p_matrix = rng.integers(low=0, high=10, size=size*size).reshape(size,size) + 1
                    splits_list = list(p2p_matrix[rank])
                    recv_splits_list = list(p2p_matrix[:, rank])
                    recv_splits_lists.append(recv_splits_list)
                    p2p_matrices.append(p2p_matrix)
                    print(f'rank={rank}/{size}, split={splits_list}, recv_split={recv_splits_list}, matrix={p2p_matrix.tolist()}', flush=True)
                    with tf.device("/cpu:0"):
                        splits = tf.constant(splits_list, dtype=idx_dtype)
                        recv_splits = tf.constant(recv_splits_list, dtype=idx_dtype)
                        tensor = tf.ones([sum(splits_list), vector_dim], dtype=dtype) * (rank + 1)
                        name = f'fused_slot_{slot}'
                        result = bps.alltoall_fused(tensor, splits=splits, recv_splits=recv_splits,
                                                    name=name, group=niter % 2, limit=num_slots)
                        outputs.append(result)

                with tf.compat.v1.Session() as sess:
                    results = sess.run(outputs)

                # verify results
                for slot in range(num_slots):
                    index = 0
                    result = results[slot]
                    for i in range(size):
                        begin = index
                        end = index + recv_splits_lists[slot][i]
                        subset = result[begin:end]
                        val = i + 1
                        assert np.sum(subset != val) == 0, (subset, val, p2p_matrices[slot][i], result)
                        index = end
                        print(f'slot {slot} rank {i} is correct, shape={subset.shape}, split={p2p_matrices[slot][i]}', flush=True)
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
                result = bps.push_pull(tensor, name=f'test_allreduce_iter_{niter % 10}')
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

tests.test_allreduce()

if is_direct_resp == 2:
    tests.test_all2all()
    tests.test_all2all(compression=bps.Compression.fp16)
    tests.test_all2all_no_recv_splits()
    tests.test_all2all_no_recv_splits(compression=bps.Compression.fp16)

if is_direct_resp == 0:
    tests.test_all2all_cpu2gpu()
    tests.test_all2all_benchmark()
    tests.test_all2all_benchmark(dst_gpu=True, src_gpu=False)
    tests.test_all2all_benchmark(dst_gpu=False, src_gpu=True)
    tests.test_all2all_benchmark(dst_gpu=True, src_gpu=True)

time.sleep(1)
