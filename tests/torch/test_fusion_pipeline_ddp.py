# Copyright 2021 Bytedance Inc. All Rights Reserved.
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

import torch
import math, sys
import numpy as np
from torch.nn.modules import linear
import torch.optim as optim
from utils import LinearRegression, check_weight, check_grad
import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='warmup test')
parser.add_argument("--warmup_iter", type=int, default=0,
                    help="number of iterations for pipesgd warmup")
args = parser.parse_args()

# initialization
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
rank, size = dist.get_rank(), dist.get_world_size()
print(f"running on rank={rank}")

probe_steps = int(os.getenv("BYTEPS_BUCKET_PROBE_STEPS", 5))

device_id = local_rank

# prepare raw data
x_np = np.array([1.0, 1.0, 1.0], dtype=np.float)
w_np = np.array([1.0, 1.0, 1.0], dtype=np.float)
y_np = np.array([2.0], dtype=np.float)
learning_rate = 0.01

# define numpy model
linear_regression = LinearRegression(weight=w_np, lr=learning_rate,
                                     warmup_iter=args.warmup_iter)

# move tensor to GPU
x = torch.from_numpy(x_np).type(torch.float32).to(device_id)
y = torch.from_numpy(y_np).type(torch.float32).to(device_id)

# Use the nn package to define our model and loss function.
model = torch.nn.Linear(3, 1, bias=False).cuda()
with torch.no_grad():
    model.weight.fill_(1.0)

model = DDP(model, device_ids=[local_rank])
loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print(f'x = {x}, y = {y}, x.dtype = {x.dtype}, y.dtype = {y.dtype}')

iteration = 20
for i in range(iteration):
    optimizer.zero_grad()
    linear_regression.zero_grad()

    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    loss.backward()
    linear_regression.backward(x_np, y_np, start_step=probe_steps)

    check_weight(model.module, linear_regression, rank, i)
    optimizer.step()
    skip_step = max(probe_steps - 1, args.warmup_iter)
    linear_regression.step(should_skip=(i == skip_step))
    check_grad(model.module, linear_regression, rank, i, start_step=skip_step)

print('All good!')
