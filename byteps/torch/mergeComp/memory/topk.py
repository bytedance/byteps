import torch
import sys
sys.path.append("../..")
from mergeComp import Memory
import byteps.torch as bps


class TopKMemory(Memory):
    def __init__(self, beta=0.9, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma
        self.zeros = {}


    def compensate(self, tensor, name):
        if name in self.residuals:
            tensor = tensor + self.beta*self.residuals[name]
        return tensor


    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        values, indices = tensor_compressed
        if name not in self.zeros:
            self.zeros[name] = torch.zeros_like(values)
        self.residuals[name] = tensor.scatter_(0, indices.type(torch.int64), self.zeros[name])
