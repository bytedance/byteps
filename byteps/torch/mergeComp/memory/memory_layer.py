import torch
from abc import ABC, abstractmethod

import byteps.torch as bps
import sys
sys.path.append("../..")
from mergeComp import Memory
import time


class MemoryLayer(Memory):
    def __init__(self, named_parameters, lr=1e-3, wd=5e-4):
        self.world_size = bps.size()
        self.tensor_num = 0
        self.lr = lr
        self.wd = wd
        self.fusion_num = 0
        self.param = {}
        self.grad = {}
        self.reduction = {}
        self.velocities = {}
        self.momentum = {}
        self.shapes = {}

        self.initialize(named_parameters)


    def initialize(self, named_parameters):
        for name, p in named_parameters:
            if p.requires_grad:
                numel = p.numel()
                if self.tensor_num == 0:
                    self.first_tensor_name = name
                self.tensor_num += 1
                self.shapes[name] = p.grad.size()
                self.param[name] = p.grad
                self.grad[name] = torch.zeros(numel).cuda()
                self.reduction[name] = torch.zeros(numel).cuda()
                self.velocities[name] = torch.zeros(numel).cuda()
                self.momentum[name] = torch.zeros(numel).cuda()


    def get_grad(self, name):
        return self.grad[name]


    def get_momentum(self, name):
        return self.momentum[name]


    def get_velocity(self, name):
        return self.velocities[name]


    def get_reduction(self, name):
        return self.reduction[name]


    @abstractmethod
    def compensate(self, tensor, name):
        raise NotImplemented("compensate was not implemented.")


    def pool_compensate(self, tensor, name):
        self.get_grad(name).copy_(tensor.view(-1))
        self.compensate(None, name)
        return self.get_velocity(name)


    @abstractmethod
    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        raise NotImplemented("update was not implemented.")


    def pool_update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        if tensor_compressed is None:
            return

        self.update(tensor, name, compressor, tensor_compressed, ctx)


    def reduce(self, ctx, name):
        pass


    def aggregate(self, tensors):
        return tensors

    def update_lr(self, lr):
        self.lr = lr


    def add_tensor(self, name, size):
        pass


    def skip_step(self):
        return False

    @torch.no_grad()
    def pool_step(self, name):
        grad = self.param[name]
        shape = self.shapes[name]
        tensor_updates = self.get_reduction(name)
        with torch.no_grad():
            grad.set_(tensor_updates.view(shape))