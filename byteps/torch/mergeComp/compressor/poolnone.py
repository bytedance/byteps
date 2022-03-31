import torch
import sys
sys.path.append("../..")
from mergeComp import Compressor

class PoolNoneCompressor(Compressor):
    """Default no-op compression."""
    def __init__(self):
        super().__init__()
        self.name = "PoolNone"
        self.quantization = False
        self.sparsification = False

    def compress(self, tensor, name, signsgd_unit_size=8, alltoall_nodes=1):
        ctx = (name, tensor.numel())
        return [tensor, tensor.abs().mean().reshape((1,))], ctx

    def decompress(self, tensors, ctx, alltoall=False):
        return tensors[0]
