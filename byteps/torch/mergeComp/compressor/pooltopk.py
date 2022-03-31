import torch
import sys
from math import ceil
sys.path.append("../..")
from mergeComp import Compressor


class PoolTopKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__()
        self.name = "PoolTopK"
        self.quantization = False
        self.sparsification = True
        self.compress_ratio = compress_ratio


    def sparsify(self, tensor, numel=0, k=0):
        if numel == 0:
            numel = tensor.numel()
        if k == 0:
            k = ceil(numel * self.compress_ratio)

        _, indices = torch.topk(tensor.abs(), k)
        values = tensor[indices]
        return values, indices.type(torch.int32)


    def compress(self, tensor, name, alltoall_nodes=1):
        numel = tensor.numel()
        num_elem_per_node = numel

        if alltoall_nodes == 1:
            tensors = self.sparsify(tensor)
            ctx = (name, tensor.numel(), num_elem_per_node, ceil(numel * self.compress_ratio))
            return tensors, ctx

        # for alltoall operation, we ensure all partitions have the same number of gradients after compression
        num_elem_per_node = ceil(numel / alltoall_nodes)
        k = ceil(num_elem_per_node * self.compress_ratio)
        values_list, indices_list = [], []
        for i in range(alltoall_nodes):
            start, end = i * num_elem_per_node, (i+1) * num_elem_per_node
            if i == alltoall_nodes - 1:
                end = numel
            values, indices = self.sparsify(tensor[start:end], end-start, k)
            values_list.append(values)
            indices_list.append(indices)
        tensors = torch.stack(values_list).reshape(-1), torch.stack(indices_list).reshape(-1)
        ctx = (name, numel, num_elem_per_node, k)
        return tensors, ctx


    def decompress(self, tensors, ctx, alltoall=False):
        name, numel, num_elem_per_node, _ = ctx
        values, indices = tensors
        
        if alltoall:
            # in alltoall compression, num_elem_per_node on the last GPU could be smaller than numel//alltoall_nodes
            # further operations is needed when torch.cat() is called
            tensor_decompressed = torch.zeros(num_elem_per_node, dtype=values.dtype, device=values.device)
        else:
            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)

        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        return tensor_decompressed