import torch

from mergeComp_dl.torch import Compressor
from mergeComp_dl.torch.util import packbits, unpackbits


class PoolOneBitCompressor(Compressor):
    def __init__(self):
        super().__init__()
        self.name = "PoolOneBit"
        self.quantization = False


    def compress(self, tensor, name, ctx, server=False):
        numel = tensor.numel()

        mask0 = tensor < 0
        sum0 = torch.sum(tensor[mask0])
        num0 = torch.sum(mask0).float()
        mean0 = sum0 / num0 if num0 > 0 else sum0
        mean0 = mean0.reshape((1,))

        mask1 = ~mask0
        sum1 = torch.sum(tensor[mask1])
        num1 = numel - num0
        mean1 = sum1 / num1 if num1 > 0 else sum1
        mean1 = mean1.reshape((1,))

        means = torch.cat((mean0, mean1))

        int8_tensor, size = packbits(mask0)
        tensor_compressed = int8_tensor, means

        ctx = (name, numel)
        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx, server=False):
        int8_tensor, means = tensor_compressed
        mean0, mean1 = means[0], means[1]
        name, numel = ctx

        uint8_tensor = unpackbits(int8_tensor, numel)

        tensor_decompressed = uint8_tensor * mean0 + ~uint8_tensor * mean1
        return tensor_decompressed