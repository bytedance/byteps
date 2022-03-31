import torch
import sys
sys.path.append("../..")
from mergeComp import Compressor
from mergeComp.util import packbits, unpackbits


class PoolSignSGDCompressor(Compressor):
    def __init__(self):
        super().__init__(average=False)
        self.name = "PoolSignSGD"
        self.quantization = True
        self.sparsification = False
        self.unit_size = 8
        self.zeros = torch.zeros([1024], dtype=torch.bool, device=torch.cuda.current_device())


    def get_scalar(self, tensor):
        return None


    def compress(self, tensor, name, alltoall_nodes=1, scalar=None):
        numel = tensor.numel()

        sign_encode = tensor >= 0
        mean = scalar
        if mean is None:
            mean = tensor.abs().mean().reshape((1,))

        # for alltoall operation, we ensure all partitions have the same number of gradients after compression
        # our solution is to add padding
        unit_size = alltoall_nodes * self.unit_size
        padding = numel % unit_size
        if padding > 0:
            padding = unit_size - padding
            sign_encode = torch.cat([sign_encode, self.zeros[:padding]], dim=0)
        packed_tensor = packbits(sign_encode)
        tensor_compressed = packed_tensor, mean
        numel_per_node = (numel + padding) // alltoall_nodes
        # note that after padding, the compressed message in the last node has some additional zero values
        # we will get rid of them after decompress all the messages
        ctx = (name, numel, numel_per_node)
        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx, alltoall=False):
        """Decoding the signs to float format """
        packed_tensor, mean = tensor_compressed
        _, numel, numel_per_node = ctx

        if alltoall:
            sign_decode = unpackbits(packed_tensor, numel_per_node)
        else:
            sign_decode = unpackbits(packed_tensor, numel)
        sign_decode = sign_decode.type(torch.float32) * 2 - 1

        return sign_decode


    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = sum(tensors)
        sign = agged_tensor >= 0
        agged_tensor = sign * 2.0 - 1.0
        return agged_tensor