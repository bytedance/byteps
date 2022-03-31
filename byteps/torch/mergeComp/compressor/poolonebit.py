import torch
import sys
sys.path.append("../..")
from mergeComp import Compressor
from mergeComp.util import packbits, unpackbits


class PoolOneBitCompressor(Compressor):
    def __init__(self):
        super().__init__()
        self.name = "PoolOneBit"
        self.quantization = True
        self.sparsification = False
        self.zeros = torch.zeros([1024], dtype=torch.bool, device=torch.cuda.current_device())


    def get_scalar(self, tensor):
        mask0 = tensor < 0
        sum0 = torch.sum(tensor[mask0])
        num0 = torch.sum(mask0).float()
        mean0 = sum0 / num0 if num0 > 0 else sum0
        mean0 = mean0.reshape((1,))

        mask1 = ~mask0
        sum1 = torch.sum(tensor[mask1])
        num1 = tensor.numel() - num0
        mean1 = sum1 / num1 if num1 > 0 else sum1
        mean1 = mean1.reshape((1,))

        return torch.cat((mean0, mean1))


    def compress(self, tensor, name, signsgd_unit_size=8, alltoall_nodes=1, scalar=None):
        numel = tensor.numel()
        means = scalar
        mask0 = tensor < 0

        if means == None:
            means = self.get_scalar(tensor)

        # for alltoall operation, we ensure all partitions have the same number of gradients after compression
        # our solution is to add padding
        unit_size = alltoall_nodes * signsgd_unit_size
        padding = numel % unit_size
        if padding > 0:
            padding = unit_size - padding
            mask0 = torch.cat([mask0, self.zeros[:padding]], dim=0)
        packed_tensor = packbits(mask0, unit_size=signsgd_unit_size)
        tensor_compressed = packed_tensor, means
        numel_per_node = (numel + padding) // alltoall_nodes
        ctx = (name, numel, numel_per_node)
        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx, alltoall=False):
        packed_tensor, means = tensor_compressed
        mean0, mean1 = means[0], means[1]
        name, numel, numel_per_node = ctx

        if alltoall:
            sign_decode = unpackbits(packed_tensor, numel_per_node)
        else:
            sign_decode = unpackbits(packed_tensor, numel)
        sign_decode = unpackbits(packed_tensor, numel)

        tensor_decompressed = sign_decode * mean0 + ~sign_decode * mean1
        return tensor_decompressed