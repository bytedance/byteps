import argparse
import os
import numpy as np
import time

import torch
import byteps.torch as bps


parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda

bps.init()
my_rank = bps.rank()
print("xxxx python myrank ", my_rank)
if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(my_rank)


size = 1024000
half_size = size // 2
for ii in range(1):
    grad = torch.ones(size, dtype=torch.float32).cuda()
    grad[:half_size].mul_(4*my_rank + 4)

    print("before push pull", grad[:half_size].mean(), grad[half_size:].mean())
    #handle = bps.byteps_push_pull(grad, average=True, name="test")
    #grad = bps.synchronize(handle)
    grad = bps.intra_push(grad, average=False, name="test")
    print("after push pull", grad[:size//2].mean(), grad[size//2:].mean())