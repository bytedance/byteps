import torch

#from extensions.cuda.functions import accumulate
from .memory_pool import MemoryPool


class PoolDgcMemory(MemoryPool):
    def __init__(self, named_parameters, lr=1e-3, momentum=0.2, fusion_num=2, gradient_clipping=False, momentum_masking=True):
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.momentum_masking = momentum_masking
        super().__init__(named_parameters, fusion_num, lr=lr)
        self.iterations = -1


    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        # https://github.com/synxlin/deep-gradient-compression/blob/master/dgc/memory.py
        grad = self.get_grad(name)
        #if self.gradient_clipping:
        #    tensor_squ_sum = torch.sum(grad * grad)
        #    clipping_val = torch.sqrt(allreduce_(tensor_squ_sum, average=True, name=name))
        #    grad = grad.clamp(-clipping_val, clipping_val)
        mmt = self.get_momentum(name)
        vec = self.get_velocity(name)

        if self.momentum_masking:
            mmt.mul_(self.momentum).add_(grad)
            vec.add_(mmt)
        else:
            vec.mul_(self.momentum).add_(grad)


    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        mask = ctx[1]
        not_mask = ~mask

        values, indices = tensor_compressed
        indices_int64 = indices.type(torch.int64)

        if self.momentum_masking:
            mmt = self.get_momentum(name)
            mmt.copy_(mmt * not_mask)

        vec = self.get_velocity(name)
        vec.copy_(vec * not_mask)


    def reduce(self, ctx, name):
        reduction = self.get_reduction(name)
        reduction.set_(sum(ctx))