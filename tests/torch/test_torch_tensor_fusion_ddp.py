from __future__ import print_function

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import copy
from torchvision import models
import torch.nn as nn
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

torch.backends.cudnn.deterministic = True
torch.manual_seed(0) # sets the seed for generating random numbers.
torch.cuda.manual_seed(0) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
torch.cuda.manual_seed_all(0) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
try:
    torch.use_deterministic_algorithms(True)
except AttributeError:
    torch.set_deterministic(True)

class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16*5*5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)

   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, self.num_flat_features(x))
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x

   def num_flat_features(self, x):
       size = x.size()[1:]
       num_features = 1
       for s in size:
           num_features *= s
       return num_features

# Benchmark settings
parser = argparse.ArgumentParser(description='Tests of PyTorch Fused Optimizer',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='the optimizer to test (one of SGD/Adam/Adagrad)')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5,
                    help='weight decay')
parser.add_argument('--dampening', type=float, default=0,
                    help='dampening')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='enable nesterov for sgd or not')
parser.add_argument('--half', action='store_true', default=False,
                    help='use half precision')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 of adam algorithm')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 of adam algorithm')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='term added to the denominator to improve numerical stability')
parser.add_argument('--amsgrad', action='store_true', default=False,
                    help='whether to use the AMSGrad variant of this algorithm')
parser.add_argument('--compare-apex', action='store_true', default=False,
                    help='should compare with apex')
parser.add_argument('--cpu-only', action='store_true', default=False,
                    help='run CPU tests only')

args = parser.parse_args()
use_cuda = not args.cpu_only

dist.init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
if use_cuda:
    torch.cuda.set_device(local_rank)

# Set up standard model.
model1 = LeNet5()
model2 = copy.deepcopy(model1)

if use_cuda:
    model1.cuda()
    model2.cuda()

model1 = DDP(model1, device_ids=[local_rank], bucket_cap_mb=0)
model2 = DDP(model2, device_ids=[local_rank])

optimizer1, optimizer2 = None, None
if args.optimizer == 'SGD':
    compareSGD = torch.optim.SGD
    optimizer1 = compareSGD(
                            model1.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            dampening=args.dampening,
                            nesterov=args.nesterov)
    optimizer2 = compareSGD(
                            model2.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            dampening=args.dampening,
                            nesterov=args.nesterov)
elif args.optimizer == 'Adam':
    compareAdam = torch.optim.Adam
    optimizer1 = compareAdam(
                        model1.parameters(),
                        lr=args.lr,
                        betas=(args.beta1, args.beta2),
                        eps=args.eps,
                        weight_decay=args.weight_decay,
                        amsgrad=args.amsgrad)
    optimizer2 = compareAdam(
                            model2.parameters(),
                            lr=args.lr,
                            betas=(args.beta1, args.beta2),
                            eps=args.eps,
                            weight_decay=args.weight_decay,
                            amsgrad=args.amsgrad)
elif args.optimizer == 'Adagrad':
    compareAdagrad = torch.optim.Adagrad
    optimizer1 = compareAdagrad(
                            model1.parameters(),
                            lr=args.lr,
                            eps=args.eps,
                            weight_decay=args.weight_decay)
    optimizer2 = compareAdagrad(
                            model2.parameters(),
                            lr=args.lr,
                            eps=args.eps,
                            weight_decay=args.weight_decay)
else:
    assert False, f'{args.optimizer} is not a valid optimizer type'


# Set up fake data
datasets = []
with torch.no_grad():
    for _ in range(100):
        data = torch.rand(args.batch_size, 1, 28, 28)
        target = torch.LongTensor(args.batch_size).random_() % 10
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        datasets.append(data)
data_index = 0

def optim_step_and_get_time(optimizer):
    t1 = time.time()
    optimizer.step()
    t2 = time.time()
    # return micro sec
    return (t2-t1) * 1e6

def is_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1.data, p2.data, atol=1e-3):
            how_many_diff = p1.data.ne(p2.data).sum().cpu()
            print(f'param 1: {p1} \nparam 2: {p2}')
            print(f'====\ntensor size: {p1.data.size()}, how many different: {how_many_diff}\n====')
            return False
    return True

assert is_equal(model1, model2), 'init model not equal'


for t in range(args.num_iters):
    data1 = datasets[data_index % len(datasets)]
    data2 = datasets[data_index % len(datasets)]
    data_index += 1
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    output1 = model1(data1)
    output2 = model2(data2)
    loss1 = F.cross_entropy(output1, target)
    loss2 = F.cross_entropy(output2, target)
    loss1.backward()
    loss2.backward()
    time1 = optim_step_and_get_time(optimizer1)
    time2 = optim_step_and_get_time(optimizer2)
    assert is_equal(model1, model2), f'pass iter {t}, result not equal!'
    if dist.get_rank() == 0:
        print(f'pass iter {t}, result equal')
