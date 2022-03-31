import torch

from mergeComp_dl.torch import Compressor
from mergeComp_dl.torch.util import packbits, unpackbits


class PoolEFSignSGDCompressor(Compressor):
    def __init__(self):
        super().__init__()
        self.name = "EFSignSGD"
        self.quantization = True


    def compress(self, tensor, name, ctx, server=False):
        """Encoding and compressing the signs """
        numel = tensor.numel()

        sign_encode = tensor >= 0
        mean = tensor.abs().mean().reshape((1,))

        int8_tensor, size = packbits(sign_encode)
        tensor_compressed = int8_tensor, mean

        ctx = (name, numel)
        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx, server=False):
        """Decoding the signs to float format """
        int8_tensor, mean = tensor_compressed
        mean = mean[0]
        name, numel = ctx

        sign_decode = unpackbits(int8_tensor, numel)
        sign_decode = sign_decode.type(torch.float32) * 2 - 1
        return mean * sign_decode
