import torch
import math
import sys
sys.path.append("../..")
from mergeComp import Compressor
from mergeComp.util import packbits, unpackbits


class PoolSignumCompressor(Compressor):
    def __init__(self, momentum):
        super().__init__(average=False)
        self.name = "PoolSignNum"
        self.quantization = True
        self.momentum = momentum
        self.momentums = {}


    def get_scalar(self, tensor):
        return None


    def compress(self, tensor, name, scalar=None):
        """Encoding and compressing the signs """
        numel = tensor.numel()
        mean = tensor.abs().mean().reshape((1,))

        # update tensor by momentum
        if name in self.momentums:
            tensor = (1.0 - self.momentum) * tensor + self.momentum * self.momentums[name]
        self.momentums[name] = tensor
        sign_encode = tensor >= 0

        int8_tensor = packbits(sign_encode)
        tensor_compressed = int8_tensor, mean

        ctx = (name, numel)
        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx):
        """Decoding the signs to float format """
        int8_tensor, _ = tensor_compressed
        name, numel = ctx

        sign_decode = unpackbits(int8_tensor, numel)
        return sign_decode.type(torch.float32) * 2 - 1


    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = sum(tensors)
        agged_tensor = agged_tensor >= 0
        agged_tensor = agged_tensor * 2.0 - 1.0
        return [agged_tensor]


    def clean(self):
        self.momentums = {}