import torch
import sys
sys.path.append("../..")
from mergeComp import Memory


class ResidualMemory(Memory):
    def __init__(self, beta=0.9, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma


    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + tensor
        return tensor


    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        self.residuals[name] = tensor - tensor_decompressed
