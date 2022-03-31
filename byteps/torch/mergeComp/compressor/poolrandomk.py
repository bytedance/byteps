import torch
import random
import sys
import numpy
from math import ceil
import byteps.torch as bps
sys.path.append("../..")
from mergeComp import Compressor


class PoolRandomKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.name = "PoolDGC"
        self.quantization = False
        self.sparsification = True
        self.compress_ratio = compress_ratio


    def _sparsify_randomk(self, tensor, numel=0, k=0):
        if numel == 0:
            numel = tensor.numel()
        if k == 0:
            k = ceil(numel * self.compress_ratio)
        indices = torch.randint(0, numel, (k,), device=tensor.device, dtype=torch.int64)
        values = tensor[indices]
        return values, indices.type(torch.int32)


    def _sparsify(self, tensor, alltoall_nodes=1):
        numel = tensor.numel()

        if alltoall_nodes > 1:
            num_elem_per_node = ceil(numel / alltoall_nodes)
            k = ceil(num_elem_per_node * self.compress_ratio)
            values_list, indices_list = [], []
            for i in range(alltoall_nodes):
                start, end = i * num_elem_per_node, (i+1) * num_elem_per_node
                if i == alltoall_nodes - 1:
                    end = numel
                values, indices = self._sparsify_randomk(tensor[start:end], end-start, k)
                values_list.append(values)
                indices_list.append(indices)
            values, indices = torch.stack(values_list).reshape(-1), torch.stack(indices_list).reshape(-1)
            return values, indices, num_elem_per_node

        num_selects = int(numel * self.compress_ratio)
        indices = torch.randint(0, numel, (num_selects,), device=tensor.device, dtype=torch.int64)
        values = tensor[indices]
        return values, indices.type(torch.int32), numel


    def compress(self, tensor, name, alltoall_nodes=1):
        numel = tensor.numel()

        values, indices, num_elem_per_gpu = self._sparsify(tensor, alltoall_nodes)
        tensor_compressed = values, indices
        ctx = (name, numel, num_elem_per_gpu)

        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx, alltoall=False):
        name, numel, numel_per_gpu = ctx
        values, indices = tensor_compressed
        if alltoall:
            tensor_decompressed = torch.zeros(numel_per_gpu, dtype=values.dtype, device=values.device)
        else:
            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        return tensor_decompressed