from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
from byteps.torch.ops import push_pull_async_inplace, poll, synchronize
import byteps.torch as bps
import time
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch BytePS Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-warmup', type=int, default=10,
                    help='number of warm-up steps that don\'t count towards benchmark')
parser.add_argument('--num-iters', type=int, default=1000,
                    help='number of benchmark iterations')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--no-wait', type=bool, default=True,
                    help='wait for other worker request first')
parser.add_argument('--gpu', type=int, default=-1,
                    help='use a specified gpu')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

bps.init()

# BytePS: pin GPU to local rank.
if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
else:
    torch.cuda.set_device(bps.local_rank())

cudnn.benchmark = True


def log(s, nl=True):
    if bps.rank() != 0:
        return
    print(s, end='\n' if nl else '')


def benchmark(tensor, average, name):
    if not args.no_wait and bps.rank() == 0:
        # let other workers submit push-pull request first
        time.sleep(0.01)
    start = time.time()
    handle = push_pull_async_inplace(tensor, average, name)
    while True:
        if poll(handle):
            synchronize(handle)
            break
    end = time.time()
    return (end - start) * 1000


log('Number of GPUs: %d' % (bps.size()))

# Benchmark
log('Running benchmark...')

log('size (Byte)    avg. time (ms)    std.dev (ms)')
for i in range(10):
    size = 10**i
    data = torch.rand(size, dtype=torch.float32)
    if args.cuda:
        data = data.cuda()
    # warm up
    for j in range(args.num_warmup):
        benchmark(tensor=data, average=True, name=str(i))
    # timeit
    durations = []
    for j in range(args.num_iters):
        t = benchmark(tensor=data, average=True, name=str(i))
        durations.append(t)
    avg = np.mean(durations)
    std = np.std(durations)

    log('%d    %s    %s' % (4*size, '%.3f'%avg, '%.3f'%std))

log('End benchmark.')
