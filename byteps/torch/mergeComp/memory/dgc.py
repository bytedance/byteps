import torch
import sys
sys.path.append("../..")
from mergeComp import Memory
import byteps.torch as bps


class DgcMemory(Memory):
    def __init__(self, momentum=0.9, gradient_clipping=False):
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.gradients = {}
        self.residuals = {}


    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        # https://github.com/synxlin/deep-gradient-compression/blob/master/dgc/memory.py
        if self.gradient_clipping:
            tensor_squ_sum = torch.sum(tensor * tensor)
            clipping_val = torch.sqrt(bps.byteps_push_pull(tensor_squ_sum, average=True, name=name))
            tensor = tensor.clamp(-clipping_val, clipping_val)

        if name in self.residuals:
            self.residuals[name] = self.momentum * self.residuals[name] + tensor
        else:
            self.residuals[name] = tensor
        
        if name in self.gradients:
            self.gradients[name] += self.residuals[name]
            tensor = self.gradients[name]
        else:
            self.gradients[name] = tensor
        return tensor



    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        mask = ctx[1]
        not_mask = ~mask

        temp = self.residuals[name] * not_mask
        self.residuals[name] = temp
        temp = self.gradients[name] * not_mask
        self.gradients[name] = temp