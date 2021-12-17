from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import torch
import os
import numpy as np
import time

import argparse
parser = argparse.ArgumentParser(description='PyTorch tests')
parser.add_argument('--backend', type=str, default='byteps')
parser.add_argument('--iter', type=int, default=250)

args = parser.parse_args()
if args.backend == 'byteps':
    import byteps.torch as bps
    bps.init(lazy=False)
else:
    import horovod.torch as bps
    bps.init()


print(f'loading byteps from {bps.__file__}')

args.iter = int(os.environ.get('TEST_NUM_ITER', args.iter))
has_gpu = os.environ.get('TEST_GPU', '1') == '1'

class TorchTests:
    """
    Tests for ops in byteps.torch
    """
    def __init__(self):
        self.rank = bps.rank()
        self.size = bps.size()
        print(bps.__file__)

    def current_context(self):
        if has_gpu:
            torch.cuda.set_device(bps.local_rank())
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def random(self, seed, size, dtype, device, requires_grad=False):
        torch.manual_seed(seed)
        return torch.rand(size=size, dtype=dtype, device=device, requires_grad=requires_grad)

    def validate_allgather(self, gathered, shape, dtype, device):
        golden = [self.random(i, shape, dtype, device)
            for i in range(self.size)]
        golden = torch.cat(golden, 0)
        max_difference = torch.max(torch.abs(gathered - golden))
        threshold = 0

        assert max_difference <= threshold, "bps.allgather produced incorrect results"
        print(f"bps.allgather test success!")

    def test_byteps_allgather(self):
        """Test on GPU that the allgather correctly runs on 1D, 2D, 3D tensors."""
        dtypes = [torch.float32]
        dims = [1, 2, 3]
        device = self.current_context()

        for iter in range(args.iter):
            for dtype, dim in itertools.product(dtypes, dims):
                shape = [10] * dim
                tensor = self.random(self.rank, shape, dtype, device)
                gathered = bps.allgather(tensor, name=f'allgather_{dtype}_{dim}_{device}_iter_{iter}')
                self.validate_allgather(gathered, shape, dtype, device)

    def validate_allreduce(self, reduced, shape, dtype, device):
        golden = [self.random(i, shape, dtype, device)
            for i in range(self.size)]
        golden = torch.stack(golden)
        golden = torch.sum(golden, 0)
        max_difference = torch.max(torch.abs(reduced - golden))
        if self.size <= 3 or dtype in [torch.int32, torch.int64]:
            threshold = 0
        elif self.size < 10:
            threshold = 1e-4
        elif self.size < 15:
            threshold = 5e-4

        assert max_difference <= threshold, "bps.allreduce produced incorrect results"
        print(f"bps.allreduce test success!")

    def test_byteps_mix_allgather_allreduce(self):
        """Test on GPU that the allgather/allreduce correctly run on 1D, 2D, 3D tensors."""
        dtypes = [torch.float32]
        dims = [1, 2, 3]
        device = self.current_context()
        for iter in range(args.iter):
            for dtype, dim in itertools.product(dtypes, dims):
                shape = [10] * dim
                allgather_input = self.random(self.rank, shape, dtype, device)
                allreduce_input = self.random(self.rank, shape, dtype, device)

                allgather_name = f'allgather_{dtype}_{dim}_{device}_iter_{iter}'
                allreduce_name = f'allreduce_{dtype}_{dim}_{device}_iter_{iter}'

                gathered = bps.allgather(allgather_input, name=allgather_name)
                reduced = bps.push_pull(allreduce_input, average=False, name=allreduce_name)

                self.validate_allgather(gathered, shape, dtype, device)
                self.validate_allreduce(reduced, shape, dtype, device)

    def validate_allgather_autograd(self, grad, shape, dtype, device):
        golden = [self.random(i, shape, dtype, device)
            for i in range(self.size)]
        golden = torch.cat(golden, 0) * 2.0
        golden = golden[self.rank * shape[0] : (self.rank + 1) * shape[0]]
        max_difference = torch.max(torch.abs(grad - golden))
        threshold = 5e-4
        assert max_difference <= threshold, "bps.allgather autograd produced incorrect results"
        print(f"bps.allgather autograd test success!")

    def test_allgather_autograd(self):
        """Test of allgather autograd."""
        device = self.current_context()
        dtype = torch.float32
        for iter in range(args.iter):
            shape = [10]
            x = self.random(self.rank, shape, dtype, device, requires_grad=True)
            name = f'allgather_{device}_iter_{iter}'
            y = bps.allgather(x, name=name)
            loss = y * y
            loss_grad = torch.ones(loss.shape, dtype=dtype, device=device)
            # loss.sum().backward()
            loss.backward(loss_grad)
            self.validate_allgather_autograd(x.grad, shape, dtype, device)

tests = TorchTests()
tests.test_byteps_allgather()
tests.test_byteps_mix_allgather_allreduce()
tests.test_allgather_autograd()