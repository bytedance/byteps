from mergeComp_dl.torch import Compressor


class PoolNoneCompressor(Compressor):
    """Default no-op compression."""
    def __init__(self):
        super().__init__()
        self.name = "PoolNone"
        self.quantization = False

    def compress(self, tensor, name, start=None, server=False):
        ctx = (name, tensor.numel())
        return [tensor], ctx

    def decompress(self, tensors, ctx, server=False):
        return tensors[0]
