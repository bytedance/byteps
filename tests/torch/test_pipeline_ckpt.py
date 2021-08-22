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
import apex
from apex import amp
import byteps.torch as bps
from utils import LinearRegression, check_weight, check_grad

parser = argparse.ArgumentParser(description='test checkpoint')
parser.add_argument("--iterations", type=int, default=15,
                    help="number of iterations, should be an interger")
parser.add_argument("--grad_acc", type=int, default=1,
                    help="gradient accumulate steps, should be 1 or larger")
parser.add_argument("--opt_level", choices=['O0','O1', 'O2', 'O3'], default='O1',
                    help="opt level for amp")
parser.add_argument("--save_freq", type=int, default='10',
                    help="save checkpoint frequency")
parser.add_argument("--load_ckpt", action='store_true',
                    help="whether to load checkpoint")
args = parser.parse_args()

# initialization
bps.init(lazy=False)
rank, size = bps.rank(), bps.size()
device_id = bps.local_rank()
print(f"running on gpu {device_id}")

# prepare raw data
# x_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
w_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
y_np = np.array([2.0], dtype=np.float32)
rng = np.random.default_rng(seed=49)
x_np = rng.standard_normal(size=3, dtype=np.float32) + 1
learning_rate = 0.01

########################### ground truth ##########################
# define numpy model
linear_regression = LinearRegression(weight=w_np,
                                     lr=learning_rate,
                                     grad_acc=args.grad_acc)
weights_np = []
grads_np = []
for i in range(args.iterations):
    weights_np.append(linear_regression.weight.copy())
    linear_regression.zero_grad()
    for j in range(args.grad_acc):
        linear_regression.backward(x_np, y_np)
    grads_np.append(linear_regression.grad.copy())
    linear_regression.step(should_skip=(i == 0))

######################## pytorch model ###########################
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
                                     backward_passes_per_step=args.grad_acc,
                                     staleness=1)
model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

print(f" test with setting iteration = {args.iterations}, grad_acc = {args.grad_acc}")
print(f'x={x} {x.dtype}, y={y} {y.dtype}')

######################### load from checkpoint ############################
loss_scalers = apex.amp._amp_state.loss_scalers
assert len(loss_scalers) == 1, loss_scalers
loss_scaler = loss_scalers[0]
start_iter = 0
weights, grads = [], []
if args.load_ckpt:
    state_dict = torch.load(f'./model_{rank}.pt')
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    amp.load_state_dict(state_dict['amp_state'])
    start_iter = state_dict['iteration'] + 1
    print(f'start_iter = {start_iter}')

######################### training ############################
i = start_iter
while i < args.iterations:
    optimizer.zero_grad()

    for j in range(args.grad_acc):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        last_step = j == args.grad_acc - 1
        with amp.scale_loss(loss, optimizer, delay_unscale=not last_step) as scaled_loss:
            scaled_loss.backward()
            if last_step:
                optimizer.synchronize()
    
    if not loss_scaler._has_overflow:
        weights.append(model.weight.data.detach().clone())
        grads.append(model.weight.grad.detach().clone())
    with optimizer.skip_synchronize():
        optimizer.step()
    if not loss_scaler._has_overflow:
        if (i+1) % args.save_freq == 0:
            state_dict = {}
            optimizer.prepare_stale_states()
            state_dict['model'] = model.state_dict()
            state_dict['optimizer'] = optimizer.state_dict()
            state_dict['amp_state'] = amp.state_dict()
            state_dict['iteration'] = i
            torch.save(state_dict, f'./model_{rank}.pt')

        i += 1

###################### print results ###########################
if rank == 0:
    tolerance = 1e-2 * args.grad_acc
    for i in range(start_iter, args.iterations):
        print(f'iteration {i}:')
        weight_th = weights[i - start_iter].cpu().numpy()[0]
        weight_np = weights_np[i]
        grad_th = grads[i - start_iter].cpu().numpy()[0]
        grad_np = grads_np[i]
        print(f'weight_np = {weight_np}')
        print(f'weight_th = {weight_th}')
        print(f'grad_np = {grad_np}')
        print(f'grad_th = {grad_th}\n')
        np.testing.assert_allclose(weight_np, weight_th,
                                   rtol=tolerance, atol=tolerance)
        if i > 0:
            np.testing.assert_allclose(grad_np, grad_th,
                                       rtol=tolerance, atol=tolerance)

    print('All good')
