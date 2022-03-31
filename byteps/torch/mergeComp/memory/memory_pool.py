import torch
from abc import ABC, abstractmethod

import byteps.torch as bps
import sys
sys.path.append("../..")
from mergeComp import Memory
import time


class MemoryPool(Memory):
    def __init__(self, named_parameters, fusion_num=2, lr=1e-3, wd=5e-4):
        self.world_size = bps.size()
        self.tensor_by_name = {}
        self.grad_size = 0
        self.tensor_num = 0
        self.lr = lr
        self.wd = wd
        self.fusion_num = max(fusion_num, 1)

        self.param = {}
        self.bp_param_names = []
        self.compress_name = {}
        self.shapes = {}
        # from the last tensor to the first one
        self.tensor_sizes = []

        self.initialize(named_parameters)


    def initialize(self, named_parameters):
        offset = 0
        for name, p in named_parameters:
            if p.requires_grad:
                size = p.numel()
                self.grad_size += size
                self.tensor_num += 1
                self.param[name] = p.grad
                self.shapes[name] = p.grad.size()
                offset += size
                #print(size, end=", ")

        self.grads = torch.zeros(offset).cuda()
        self.momentums = torch.zeros(offset).cuda()
        self.velocities = torch.zeros(offset).cuda()
        self.reduction = torch.zeros(offset).cuda()


    def can_partition_memory(self):
        return len(self.bp_param_names) == self.tensor_num


    # TODO: handle local SGD
    def add_tensor(self, name, size):
        if (name, size) not in self.bp_param_names:
            self.bp_param_names.append((name, size))
            self.tensor_sizes.append(size)

        if self.can_partition_memory():
            self.last_tensor_name = self.bp_param_names[-1][0]
            self.set_tensor_by_name()
            self.partition()
            # print(self.bp_param_names)
            # print(self.tensor_sizes)

    def get_last_tensor_name(self):
        return self.last_tensor_name


    def set_tensor_by_name(self):
        offset = 0
        for name, size in self.bp_param_names:
            self.tensor_by_name[name] = (offset, offset+size)
            offset += size


    #partition the model with specific partition strategy
    def partition(self, partition_indices=None):
        if not self.can_partition_memory():
            print("Error: BP tensors are not initialized yet!")
            return

        if partition_indices is None:
            step = self.tensor_num // self.fusion_num
            partition_indices = [i*step for i in range(1, self.fusion_num)]

        fusion_thresholds = [i for i in partition_indices]
        fusion_thresholds.append(self.tensor_num)

        last_offset, offset = 0, 0
        last_name_offset, name_offset, last_offset = 0, 0, 0
        fusion_index = 0
        for name, size in self.bp_param_names:
            offset += size
            name_offset += 1
            if name_offset >= fusion_thresholds[fusion_index]:
                self.compress_name[name] = ((last_offset, offset), (last_name_offset, name_offset))
                last_offset = offset
                last_name_offset = name_offset
                fusion_index += 1
        if bps.local_rank() == 0:
            print(self.compress_name, flush=True)


    def clean(self):
        self.compress_name = {}

        #if hvd.rank() == 0:
            #print(self.compress_name)

    def get_grad(self, name):
        pos, _ = self.compress_name[name]
        return self.grads[pos[0]: pos[1]]


    def get_grad_tensor(self, name):
        pos = self.tensor_by_name[name]
        return self.grads[pos[0]: pos[1]]


    def get_momentum(self, name):
        pos, _ = self.compress_name[name]
        return self.momentums[pos[0]: pos[1]]


    def get_momentum_tensor(self, name):
        pos = self.tensor_by_name[name]
        return self.momentums[pos[0]: pos[1]]


    def get_velocity(self, name):
        pos, _ = self.compress_name[name]
        return self.velocities[pos[0]: pos[1]]


    def get_velocity_tensor(self, name):
        pos = self.tensor_by_name[name]
        return self.velocities[pos[0]: pos[1]]


    def get_reduction(self, name):
        pos, _ = self.compress_name[name]
        return self.reduction[pos[0]: pos[1]]


    def get_reduction_tensor(self, name):
        pos = self.tensor_by_name[name]
        return self.reduction[pos[0]: pos[1]]


    @abstractmethod
    def compensate(self, tensor, name):
        raise NotImplemented("compensate was not implemented.")


    def pool_compensate(self, tensor, name):
        grad = self.get_grad_tensor(name)
        grad.copy_(tensor.view(-1))

        if name in self.compress_name:
            # compensate once for merged tensors
            self.compensate(None, name)
            offset, _ = self.compress_name[name]
            start, end = offset
            return self.velocities[start:end]
        else:
            return None


    @abstractmethod
    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        raise NotImplemented("update was not implemented.")


    def pool_update(self, tensor, name, compressor, tensor_compressed, ctx, merged=True):
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


    def skip_step(self):
        return False
        #return self.fusion_num > 1


    def pool_step(self, name):
        offset, name_indices = self.compress_name[name]
        start, end = name_indices
        for i in range(start, end):
            _name, _ = self.bp_param_names[i]
            grad = self.param[_name]
            shape = self.shapes[_name]
            tensor_updates = self.get_reduction_tensor(_name)
            if not self.skip_step():
                with torch.no_grad():
                    grad.set_(tensor_updates.view(shape))