import tensorflow as tf
import byteps.tensorflow as bps

from .memory_pool import MemoryPool


class PoolNoneMemory(MemoryPool):
    def __init__(self, named_parameters, fusion_num=2):
        self.world_size = bps.size()
        super().__init__(named_parameters, fusion_num)

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        grad = self.get_grad(name)
        residual = self.get_velocity(name)
        residual.assign(grad)

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

    def reduce(self, ctx, name):
        reduction = self.get_reduction(name)
        #reduction -= reduction
        # TODO:for compression algorithms with allreduce, the received results have been averaged already.
        # Probably there is no need to divide c with self.world_size.
        
        #print(len(reduction), len(ctx))
        reduction.assign(ctx)