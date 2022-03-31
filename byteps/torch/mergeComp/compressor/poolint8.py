import torch
import sys
sys.path.append("../..")
from mergeComp import Compressor


class PoolInt8Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    def __init__(self):
        super().__init__()
        self.name = "PoolInt8"
        self.quantization = False

    def compress(self, tensor, name):
        """Downcasts the tensor to 8-bit."""
        dtype = tensor.dtype

        tensor_compressed = tensor
        if dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.uint8)
        ctx = (name, dtype)
        return [tensor_compressed], ctx


    def decompress(self, tensors, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_compressed = tensors[0]
        _, dtype = ctx
        tensor_decompressed = tensor_compressed
        #print("[decompress] before", ctx, torch.sum(tensor_compressed))
        if dtype.is_floating_point:
            tensor_decompressed = tensor_compressed.type(dtype)
        #print("[decompress] after", ctx, torch.sum(tensor_compressed))
        return tensor_decompressed