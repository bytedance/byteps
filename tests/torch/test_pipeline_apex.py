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
import numpy as np
import torch.optim as optim
import argparse

from apex import amp
import byteps.torch as bps
from utils import LinearRegression, check_weight, check_grad, is_nan


parser = argparse.ArgumentParser(description='apex test')
parser.add_argument("--iterations", type=int, default=10,
                    help="number of iterations, should be an interger")
parser.add_argument("--grad_acc", type=int, default=1,
                    help="gradient accumulate steps, should be 1 or larger")
parser.add_argument("--opt_level", choices=['O0','O1', 'O2', 'O3'], default='O1',
                    help="opt level for amp")
args = parser.parse_args()
grad_acc = args.grad_acc

# initialization
bps.init(lazy=False)
rank, size = bps.rank(), bps.size()
print(f"running on gpu {bps.local_rank()}")
device_id = bps.local_rank()

# prepare raw data
x_np = np.array([1.0, 1.0, 1.0], dtype=np.float)
w_np = np.array([1.0, 1.0, 1.0], dtype=np.float)
y_np = np.array([2.0], dtype=np.float)
learning_rate = 0.01

# define numpy model
linear_regression = LinearRegression(weight=w_np, 
                                    lr=learning_rate, 
                                    grad_acc=grad_acc)

# define torch model and data
x = torch.from_numpy(x_np).type(torch.float32).to(device_id)
y = torch.from_numpy(y_np).type(torch.float32).to(device_id)
model = torch.nn.Linear(3, 1, bias=False).cuda()
with torch.no_grad():
    model.weight.fill_(1.0)
loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = bps.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     backward_passes_per_step=grad_acc,
                                     staleness=1)
model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

print(f" test with setting iteration = {args.iterations}, grad_acc = {grad_acc}")
print(f' x = {x}, y = {y}, x.dtype = {x.dtype}, y.dtype = {y.dtype}')

first_update = True
for i in range(args.iterations):
    has_nan = False
    optimizer.zero_grad()
    linear_regression.zero_grad()

    for j in range(grad_acc):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        last_step = j == grad_acc - 1

        with amp.scale_loss(loss, optimizer, delay_unscale=not last_step) as scaled_loss:
            scaled_loss.backward()
            if last_step:
                optimizer.synchronize()

        # check if gradient overflows
        if is_nan(model.weight.grad):
            has_nan = True
        # we want to do backward for ground truth regardless of skip or not
        # the reason is there is no easy way to know whether gradient overflow
        # occurs or not in current step as we can also access gradient in the 
        # next iteration; therefore, we should always backward ground truth 
        # and skip update if we ever detect an invalid gradient of torch model
        linear_regression.backward(x_np, y_np)

    check_weight(model, linear_regression, rank, i, tolerance=1e-3)
    if not has_nan:
        check_grad(model, linear_regression, rank, i, tolerance=1e-3)

    with optimizer.skip_synchronize():
        optimizer.step()
    # even if skip == True, we still need to call optimizer.step
    # since apex.amp would handle gradient overflow
    # but we should skip linear_regression.step to keep up with torch model
    linear_regression.step(should_skip=first_update or has_nan)
    if not has_nan and first_update:
        first_update = False

print('All good!')
