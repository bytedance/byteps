from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import timeit
import numpy as np
import os
import byteps.torch.cross_barrier as bps

"""
This example shows how to enable barrier crossing on top of BytePS in PyTorch. Note that you can use BytePS without
crossing barrier at all.

Crossing barrier enables overlapping gradient push-pull with both backward computation and forward computation, while
maintaining correct dependencies, e.g., the forward computation of a layer will not start until the parameter of this
layer is updated. Hence it can further improves training performance beyond BytePS. See the paper
https://dl.acm.org/citation.cfm?id=3359642 for more details.

To use it, just change the import statement and add two more arugments (i.e., model, num_steps) when wrapping the Torch
optimizer, as shown below:
```
import byteps.torch.cross_barrier as bps
optimizer = bps.CrossBarrier(model, optimizer, named_parameters, compression, backward_passes_per_step, num_steps)
```
So far we support SGD, Adam and RMSprop optimizers. Please submit a ticket if you need support for
any other optimizers.

To see performance gain, the system parameters should be properly set, including BYTEPS_PARTITION_BYTES and
BYTEPS_SCHEDULING_CREDIT.
"""

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark Without Barrier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--partition', type=int, default=None,
                    help='partition size')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

bps.init()

if args.cuda:
    # BytePS: pin GPU to local rank.
    torch.cuda.set_device(bps.local_rank())

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)(num_classes=args.num_classes)

if args.cuda:
    # Move model to GPU.
    model.cuda()

# You may try one of the following optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# BytePS: (optional) compression algorithm.
compression = bps.Compression.fp16 if args.fp16_allreduce else bps.Compression.none

# Wrap Torch optimizer with CrossBarrier.
# You need to specify two additional args, i.e., model and num_steps.
# Note that we only support SGD, Adam and RMSProp optimizers so far.
optimizer = bps.CrossBarrier(model,
                                     optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     num_steps=args.num_warmup_batches + args.num_iters * args.num_batches_per_iter)

# BytePS: broadcast parameters & optimizer state.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
bps.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fake data
datasets = []
for _ in range(100):
    data = torch.rand(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    datasets.append(data)
data_index = 0


def benchmark_step():
    global data_index

    data = datasets[data_index%len(datasets)]
    data_index += 1
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


def log(s, nl=True):
    if bps.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, bps.size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
enable_profiling = args.profiler & (bps.rank() == 0)

with torch.autograd.profiler.profile(enable_profiling, True) as prof:
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)


# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (bps.size(), device, bps.size() * img_sec_mean, bps.size() * img_sec_conf))

