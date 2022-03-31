import torch
import random
import sys
import numpy
from math import ceil
import byteps.torch as bps
sys.path.append("../..")
from mergeComp import Compressor


class PoolDgcCompressor(Compressor):
    def __init__(self, compress_ratio, sample_ratio=0.001, strided_sample=True,
                 compress_upper_bound=2, compress_lower_bound=1,
                 max_adaptation_iters=10):
        super().__init__(tensors_size_are_same=False)
        self.name = "PoolDGC"
        self.quantization = False
        self.sparsification = True
        self.compress_ratio = compress_ratio
        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.sample_ratio = min(max(sample_ratio, 0.001), 1.0)
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.attributes = (self.sample_ratio, self.base_compress_ratio)
        self.zeros = {}
        self.masks = {}
        self.stride = 1 // self.compress_ratio


    def _sparsify_randomk(self, tensor, numel=0, k=0):
        if numel == 0:
            numel = tensor.numel()
        if k == 0:
            k = ceil(numel * self.compress_ratio)
        indices = torch.randint(0, numel, (k,), device=tensor.device, dtype=torch.int64)
        values = tensor[indices]
        return values, indices.type(torch.int32)


    def _sparsify(self, tensor, alltoall_nodes=1):
        sample_ratio, compress_ratio = self.attributes
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
            return values, indices, num_elem_per_node, None

        if numel < 1024*16:
            k = ceil(numel * compress_ratio)
            _, indices = torch.topk(tensor.abs(), k)
            values = tensor[indices]
            return values, indices, numel, None

        if numel <= 1024*1024:
            sample_ratio = 0.01

        num_selects = int(numel * compress_ratio // alltoall_nodes * alltoall_nodes)
        num_samples = int(numel * sample_ratio)
            
        if self.strided_sample:
            sample_stride = int(1 // sample_ratio)
            sample_start = random.randint(0, min(sample_stride, numel-1))
            samples = tensor[sample_start::sample_stride]
        else:
            samples = tensor[torch.randint(0, numel, (num_samples, ), device=tensor.device)]

        k = ceil(num_samples * compress_ratio * 1.1)
        thr = torch.min(torch.topk(samples.abs(), k, 0, largest=True, sorted=False)[0])

        if thr != thr:
            indices = torch.randint(0, numel, (num_selects,), device=tensor.device, dtype=torch.int64)
            values = tensor[indices]
            return values, indices.type(torch.int32), numel, None

        mask = tensor.abs() >= thr
        selected = mask.sum()
        if selected == 0:
            indices = torch.randint(0, numel, (num_selects,), device=tensor.device, dtype=torch.int64)
            values = tensor[indices]
            return values, indices.type(torch.int32), numel, None

        for _ in range(self.max_adaptation_iters):
            if selected < num_selects:
                thr = 0.8 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        numel_per_gpu = ceil(numel/alltoall_nodes)
        indices,  = torch.where(mask)
        if indices.numel() >= num_selects:
            indices = indices[:num_selects]  
            mask = mask[:num_selects]
        else:
            indices = torch.randint(0, numel, (num_selects,), device=tensor.device, dtype=torch.int64)
            mask = None

        values = tensor[indices]
        return values, indices.type(torch.int32), numel_per_gpu, None


    def compress(self, tensor, name, alltoall_nodes=1):
        numel = tensor.numel()

        values, indices, num_elem_per_gpu, mask = self._sparsify(tensor, alltoall_nodes)
        tensor_compressed = values, indices
        ctx = (name, numel, num_elem_per_gpu, mask)

        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx, alltoall=False):
        name, numel, numel_per_gpu, _ = ctx
        values, indices = tensor_compressed
        if alltoall:
            tensor_decompressed = torch.zeros(numel_per_gpu, dtype=values.dtype, device=values.device)
        else:
            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        return tensor_decompressed