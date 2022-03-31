import tensorflow as tf
import byteps.tensorflow as bps

from .memory_pool import MemoryPool


class PoolResidualMemory(MemoryPool):
    #TODO: tune beta and gamma to increase accurary
    def __init__(self, named_parameters, fusion_num=2, beta=0.9, gamma=1.0):
        self.beta = beta
        self.gamma = gamma
        self.world_size = hvd.size()
        super().__init__(named_parameters, fusion_num)


    def compensate(self, tensor, name):
        """vec stores the residuals"""
        grad = self.get_grad(name)
        residual = self.get_velocity(name)
        #residual.add_(grad)
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
            #reduction.add_(c)
            reduction.add_(c/self.world_size)
