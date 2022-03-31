import torch
import byteps.torch as bps


class IntraCommComp():
    # TODO: memory for both intra_compressor and inter_compressor
    def __init__(self, compressor, DDPbackend):
        self.compressor = compressor
        self.DDPbackend = DDPbackend
        self.local_size = bps.local_size()
        self.worker_num = self.DDPbackend.worker_num


    def reducescatter(self, tensor):
        if self.local_size == 1:
            return tensor

        return self.DDPbackend.intra_reduce_scatter(tensor, async_op=True)


    def allgather(self, tensors_compressed, async_op=True):
        if self.local_size == 1:
            return tensors_compressed

        # if (len(tensors_compressed) == 1):
        #     tensor = self.DDPbackend.intra_allgather(tensors_compressed[0], async_op=async_op)
        #     return torch.stack(tensor).reshape(-1)

        ret = []
        for _, tensor in enumerate(tensors_compressed):
            tensors = self.DDPbackend.intra_allgather(tensor, async_op=async_op)
            ret.append(torch.stack(tensors).reshape(-1))
        return ret


    def alltoall(self, tensors_compressed, is_topk_like=True):
        if self.local_size == 1:
            return tensors_compressed

        # if (len(tensors_compressed) == 1):
        #     tensor = self.DDPbackend.intra_alltoall(tensors_compressed[0], async_op=True)
        #     return torch.stack(tensor).reshape(-1)

        assert(len(tensors_compressed) == 2)
        ret = []
        if not is_topk_like:
            # signsgd-like compression algorithms
            # intra alltoall the compressed data
            data = self.DDPbackend.intra_alltoall(tensors_compressed[0], async_op=True)
            # intra allgather the metadata
            metadata = self.DDPbackend.intra_allgather(tensors_compressed[1], async_op=False)
        else:
            # intra alltoall the values
            data = self.DDPbackend.intra_alltoall(tensors_compressed[0], async_op=True)
            # intra alltoall the indices
            metadata = self.DDPbackend.intra_alltoall(tensors_compressed[1], async_op=True)
        
        ret.append(torch.stack(data).reshape(-1))
        ret.append(torch.stack(metadata).reshape(-1))
        return ret


    def gather(self, tensor_compressed):
        if self.local_size == 1:
            return tensor_compressed

        ret = []
        # TODO: rotate the root_node 
        for tensor in tensor_compressed:
            tensors = self.DDPbackend.intra_gather(tensor, dst=0, async_op=True)
            ret.append(torch.stack(tensors).reshape(-1))
        return ret


    def broadcast(self, tensor_compressed):
        if self.local_size == 1:
            return tensor_compressed

        ret = []
        # TODO: rotate the root_node 
        for tensor in tensor_compressed:
            tensors = self.DDPbackend.intra_broadcast(tensor, src=0, async_op=True)
            ret.append(tensors)
        return ret

        
    def broadcast_decompress(self, tensors_compressed, ctx, is_topk_like=True, nodes=1):
        if is_topk_like or nodes == 1:
            return self.compressor.decompress(tensors_compressed, ctx)
        
        tensors, metadata = tensors_compressed
        tensors, metadata = tensors.chunk(nodes), metadata.chunk(nodes)
        tensors_decompressed = []
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx, alltoall=True))

        numel = ctx[1]
        return torch.cat(tensors_decompressed, dim=0)[:numel]


    def allgather_decompress(self, tensors_compressed, ctx):
        # allgather_decompress is for alltoall + allgather
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.local_size), metadata.chunk(self.local_size)
        tensors_decompressed = []
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx, alltoall=True))

        numel = ctx[1]
        return torch.cat(tensors_decompressed, dim=0)[:numel] / self.worker_num


    def gather_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        tensors, metadata = tensors_compressed
        if is_topk_like: 
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)
            return self.compressor.decompress((tensors, metadata), ctx)
        
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.local_size), metadata.chunk(self.local_size)

        tensors_decompressed = []
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))

        return sum(tensors_decompressed)


    def alltoall_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        if is_topk_like:
            return self.compressor.decompress(tensors_compressed, ctx, alltoall=True)

        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        tensors, metadata = tensors.chunk(self.local_size), metadata.chunk(self.local_size)
        tensors_decompressed = []
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx, alltoall=True))

        return sum(tensors_decompressed)