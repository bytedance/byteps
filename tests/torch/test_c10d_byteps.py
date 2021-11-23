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

import torch.distributed as dist
import itertools
import torch
import os
import numpy as np
import time
import argparse

has_gpu = os.environ.get('TEST_GPU', '1') == '1'

class TorchTest:
    """
    Tests for ops in byteps.torch
    """

    def __init__(self, rank, size):
        self.rank = rank
        self.size = size

    def _current_context(self):
        if has_gpu:
            torch.cuda.set_device(args.local_rank)
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _random(self, seed, size, dtype, device, requires_grad=False):
        torch.manual_seed(seed)
        return torch.rand(size=size, dtype=dtype, device=device, requires_grad=requires_grad)

    def filter_supported_types(self, types):
        return types

    def test_c10d_all_reduce(self):
        """Test that the push_pull correctly sums 1D, 2D, 3D tensors."""
        size = dist.get_world_size()
        dtypes = self.filter_supported_types([torch.float32])
        dims = [1]
        ctx = self._current_context()

        for dtype, dim in itertools.product(dtypes, dims):
            for count in range(20):
                tensor = torch.ones((100, 100), device=ctx, dtype=dtype)
                dist.all_reduce(tensor)
                result = tensor
                result_np = result.cpu().numpy()
                assert np.sum(result_np != size) == 0, result_np
                print(f'test_c10_all_reduce done niter {count}', flush=True)
        print(f'DONE test_c10_all_reduce gpu={has_gpu}', flush=True)

    def validate_allgather(self, gathered, shape, dtype, device):
        golden = [self._random(i, shape, dtype, device)
            for i in range(self.size)]
        golden = torch.cat(golden, 0)
        max_difference = torch.max(torch.abs(gathered - golden))
        threshold = 0

        try:
            assert max_difference <= threshold
        except AssertionError:
            if self.rank == 0:
                print(f'gathered:\n{gathered.cpu().numpy()}\n'
                      f'golden:\n{golden.cpu().numpy()}')
            raise AssertionError(f"dist.all_gather produced incorrect results, rank={self.rank}")
        print(f"dist.all_gather test success!")


    def test_c10d_allgather(self):
        """Test on GPU that the allgather correctly runs on 1D, 2D, 3D tensors."""
        dtypes = [torch.float32]
        dims = [1, 2, 3]
        device = self._current_context()

        for niter in range(args.niter):
            for dtype, dim in itertools.product(dtypes, dims):
                shape = [10] * dim
                tensor = self._random(self.rank, shape, dtype, device)
                world_size = dist.get_world_size()
                output_tensors = [tensor.new(torch.Size(shape)) for _ in range(world_size)]
                dist.all_gather(output_tensors, tensor)
                gathered = torch.cat(output_tensors)
                self.validate_allgather(gathered, shape, dtype, device)

    def validate_allgather_autograd(self, grad, shape, dtype, device):
        golden = [self._random(i, shape, dtype, device)
            for i in range(self.size)]
        golden = torch.cat(golden, 0) * 2.0
        golden = golden[self.rank * shape[0] : (self.rank + 1) * shape[0]]
        max_difference = torch.max(torch.abs(grad - golden))
        threshold = 5e-4
        assert max_difference <= threshold, "dist.all_gather autograd produced incorrect results"
        print(f"dist.all_gather autograd test success!")


    def test_c10d_allgather_autograd(self):
        """Test of allgather autograd."""
        device = self._current_context()
        dtype = torch.float32
        for niter in range(args.niter):
            shape = [10]
            x = self._random(self.rank, shape, dtype, device, requires_grad=True)
            world_size = dist.get_world_size()
            output_tensors = [x.new(torch.Size(shape)) for _ in range(world_size)]
            dist.all_gather(output_tensors, x)
            y = torch.cat(output_tensors)
            loss = y * y
            loss_grad = torch.ones(loss.shape, dtype=dtype, device=device)
            loss.backward(loss_grad)
            self.validate_allgather_autograd(x.grad, shape, dtype, device)

    def test_c10d_barrier(self):
        ctx = self._current_context()
        for niter in range(args.niter):
            dist.barrier()
        print(f"dist.barrier test success!")

    def validate_reduce(self, result, shape, dtype, device, op):
        golden = [self._random(i, shape, dtype, device)
            for i in range(self.size)]
        golden = torch.stack(golden, 0)

        if op == dist.ReduceOp.MIN:
            golden = torch.min(golden, dim=0)[0]
        elif op == dist.ReduceOp.MAX:
            golden = torch.max(golden, dim=0)[0]
        elif op == dist.ReduceOp.SUM:
            golden = torch.sum(golden, dim=0)
        else:
            raise ValueError(f'Unsupported operation, reduce(op={op})')

        max_difference = torch.max(torch.abs(result - golden))
        threshold = 0

        try:
            assert max_difference <= threshold
        except AssertionError:
            if self.rank == 0:
                print(f'reduced:\n{result.cpu().numpy()}\n'
                      f'golden:\n{golden.cpu().numpy()}')
            raise AssertionError(f"dist.reduce produced incorrect results, rank={self.rank}")
        if self.rank == 0:
            print(f"dist.reduce(op={op}) test success!")

    def test_c10d_reduce(self, op=dist.ReduceOp.SUM):
        """Test on GPU that the allgather correctly runs on 1D, 2D, 3D tensors."""
        dtypes = [torch.float32]
        dims = [1, 2, 3]
        device = self._current_context()

        for niter in range(args.niter):
            for dtype, dim in itertools.product(dtypes, dims):
                shape = [10] * dim
                tensor = self._random(self.rank, shape, dtype, device)
                world_size = dist.get_world_size()
                input_tensors = [self._random(idx, shape, dtype, device) for idx in range(world_size)]
                dist.reduce(tensor, dst=niter % world_size, op=op)
                self.validate_reduce(tensor, shape, dtype, device, op=op)

parser = argparse.ArgumentParser(description='PyTorch disttributed tests',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--niter', type=int, default=20)
# parser.add_argument('--niter', type=int, default=1)
args = parser.parse_args()
args.local_rank = int(os.environ["LOCAL_RANK"])
assert(args.local_rank != -1)

if __name__ == '__main__':
    dist.init_process_group(backend="nccl")
    test = TorchTest(rank=dist.get_rank(), size=dist.get_world_size())
    test.test_c10d_all_reduce()
    test.test_c10d_barrier()
    test.test_c10d_allgather()
    test.test_c10d_allgather_autograd()
    test.test_c10d_reduce()
    test.test_c10d_reduce(op=dist.ReduceOp.MIN)
    test.test_c10d_reduce(op=dist.ReduceOp.MAX)
    time.sleep(1)
