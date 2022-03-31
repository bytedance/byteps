import torch

from mergeComp_dl.torch import Compressor


def sparsify(tensor, compress_ratio):
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k)
    values = tensor[indices]
    return values, indices.type(torch.int32)


def desparsify(tensors, numel):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
    tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
    return tensor_decompressed


class PoolTopKCompressor(Compressor):

    def __init__(self, compress_ratio):
        super().__init__()
        self.name = "PoolTopK"
        self.quantization = False
        self.compress_ratio = compress_ratio


    def compress(self, tensor, name, start):
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = (name, tensor.numel(), tensor.size())
        return tensors, ctx


    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        name, numel, size = ctx
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed
