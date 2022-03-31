import torch
import horovod.torch as hvd

from .memory_layer import MemoryLayer


class ResidualMemory(MemoryLayer):
    def __init__(self, named_parameters, beta=0.9, gamma=1.0):
        self.beta = beta
        self.gamma = gamma
        self.world_size = hvd.size()
        super().__init__(named_parameters)


    def compensate(self, tensor, name):
        """vec stores the residuals"""
        grad = self.get_grad(name)
        residual = self.get_velocity(name)
        residual.mul_(self.beta).add_(self.gamma*grad)


    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = self.get_velocity(name)
        residual.assign(tensor.view(-1) - tensor_decompressed)


    def reduce(self, ctx, name):
        reduction = self.get_reduction(name)
        reduction.zero_()
        for c in ctx:
            reduction.add_(c/self.world_size)