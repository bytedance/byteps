import torch
import sys
sys.path.append("../..")
from mergeComp import Memory


class EFSignSGDMemory(Memory):
    def __init__(self, lr=0.5):
        self.residuals = {}
        # the training is sensitive to lr
        # for ResNet50 + CIFAR100 + EFSignSGD/OneBit, lr = 0.5. if lr = 0.6, the loss becomes nan. If lr is smaller, the gradients become zero
        self.lr = lr


    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.lr * self.residuals[name] + tensor
        return tensor


    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        self.residuals[name] = tensor - tensor_decompressed