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

from __future__ import print_function
import numpy as np
import os

class LinearRegression:
    """LR with stale gradient for testing purpose"""
    def __init__(self, weight=None, lr=None, grad_acc=1, warmup_iter=0) -> None:
        self.weight = weight
        self.lr = lr
        self.staled_grad = np.zeros_like(self.weight)
        self.grad = np.zeros_like(self.weight)
        self.grad_acc = grad_acc
        self.step_cnt = 0
        self.warmup_iter = warmup_iter

    def step(self, should_skip=False):
        if should_skip:
            return
        self.weight -= self.lr * self.grad

    def backward(self, x=None, y=None, start_step=0):
        '''backward computation with gradient accumulation support.'''
        self.grad += 2 * x * (self.weight @ x - y)
        self.step_cnt += 1
        if self.step_cnt % self.grad_acc == 0 and self.step_cnt >= start_step:
            if (self.step_cnt / self.grad_acc) > (self.warmup_iter + 1):
                self.grad, self.staled_grad = self.staled_grad, self.grad
            else:
                self.grad, self.staled_grad = self.grad, self.grad

    def zero_grad(self):
        self.grad = np.zeros_like(self.weight)

def check_weight(th_model, np_model, rank, i, tolerance=1e-5):
    weight_th = th_model.weight.data.clone().cpu().numpy()[0]
    weight_np = np_model.weight
    if rank == 0:
        print(f'\n iter {i}: model weight = {weight_th}'
              f'\n iter {i}: numpy weight = {weight_np}\n')
    np.testing.assert_allclose(weight_np, weight_th,
                               rtol=tolerance, atol=tolerance)

def check_grad(th_model, np_model, rank, i, tolerance=1e-5, start_step=0):
    grad_th = th_model.weight.grad.clone().cpu().numpy()[0]
    grad_np = np_model.grad
    if rank == 0:
        print(f'\n iter {i}, model grad = {grad_th}'
              f'\n iter {i}, numpy grad = {grad_np}\n')
    # we start to check the grad since iter 1
    if i > start_step:
        np.testing.assert_allclose(grad_np, grad_th,
                                   rtol=tolerance, atol=tolerance)

def is_nan(grad):
    cpu_sum = float(grad.float().sum())
    if cpu_sum == float('inf') or cpu_sum == float('-inf') or cpu_sum != cpu_sum:
        return True
    return False
