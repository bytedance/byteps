import torch

from mergeComp_dl.torch import Compressor


def sparsify(tensor, compress_ratio):
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    indices = torch.randperm(numel, device=tensor.device)[:k]
    values = tensor[indices]
    return values, indices.type(torch.int32)


class PoolRandomKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__()
        self.name = "RandomK"
        self.quantization = False
        self.compress_ratio = compress_ratio


    def compress(self, tensor, name, start):
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = name, tensor.numel(), tensor.size()
        return tensors, ctx


    def decompress(self, tensors, ctx):
        name, numel, size = ctx
        values, indices = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        return tensor_decompressed
