import torch
import horovod.torch as hvd

from .memory_layer import MemoryLayer


class NoneMemory(MemoryLayer):
    def __init__(self, named_parameters):
        self.world_size = hvd.size()
        super().__init__(named_parameters)

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        grad = self.get_grad(name)
        residual = self.get_velocity(name)
        residual.copy_(grad)

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

    def reduce(self, ctx, name):
        reduction = self.get_reduction(name)
        reduction.zero_()
        for c in ctx:
            reduction.add_(c/self.world_size)
