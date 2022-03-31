import torch
import math

from mergeComp_dl.torch import Compressor
from mergeComp_dl.torch.util import pack2bits, unpack2bits


class PoolTernGradCompressor(Compressor):
    def __init__(self):
        super().__init__()
        self.name = "PoolTernGrad"
        self.quantization = True


    def compress(self, tensor, name, ctx, server=False):
        numel = tensor.numel()
        abs_gradient = tensor.abs()
        scalar = abs_gradient.max()
        sign_gradient = tensor.sign() * scalar

        try:
            rnd_sample = torch.empty_like(tensor).cuda().uniform_(0, scalar.item())
        except:
            rnd_sample = torch.zeros_like(tensor).cuda()

        sign_gradient[rnd_sample >= abs_gradient] = 0

        mask = sign_gradient.sign() > 0
        tern_tensor = sign_gradient.sign() + 1  # {-1, 0, 1} + 1
        print(tern_tensor.sum())

        int8_tensor, size = pack2bits(mask, tern_tensor)
        tensor_compressed = int8_tensor, scalar.flatten()

        ctx = (name, numel)
        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx, server=False):
        int8_tensor, scalar = tensor_compressed
        name, numel = ctx

        tern_tensor = unpack2bits(int8_tensor, numel)
        print(tern_tensor.sum())

        sign = tern_tensor.type(torch.float32) - 1  # {0, 1, 2} - 1
        return sign * scalar