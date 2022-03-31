import torch
import byteps.torch as bps


class GlobalCommComp():
    # TODO: memory for both intra_compressor and inter_compressor
    def __init__(self, compressor, DDPbackend):
        self.compressor = compressor
        self.DDPbackend = DDPbackend
        self.world_size = bps.size()


    def allgather(self, tensors_compressed, async_op=True): 
        if self.world_size == 1:
            return tensors_compressed
        
        ret = []
        for _, tensor in enumerate(tensors_compressed):
            tensors = self.DDPbackend.global_allgather(tensor, async_op=async_op)
            ret.append(torch.stack(tensors).reshape(-1))
        return ret


    def allgather_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        
        if is_topk_like:
            return self.compressor.decompress(tensors_compressed, ctx)
        tensors_decompressed = []
        
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.world_size), metadata.chunk(self.world_size)
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))

        return sum(tensors_decompressed) / self.world_size


    def alltoall(self, tensors_compressed, async_op=True):
        if self.world_size == 1:
            return tensors_compressed
        
        assert(len(tensors_compressed) == 2)
        tensor, metadata = tensors_compressed

        ret = []
        tensors = self.DDPbackend.global_alltoall(tensor, async_op=async_op)
        ret.append(torch.stack(tensors).reshape(-1))

        metas = self.DDPbackend.global_allgather(metadata, async_op=async_op)
        ret.append(torch.stack(metas).reshape(-1))

        return ret

    
    def alltoall_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        
        if is_topk_like:
            return self.compressor.decompress(tensors_compressed, ctx, alltoall=True)
        tensors_decompressed = []
        
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.world_size), metadata.chunk(self.world_size)
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx, alltoall=True))

        return sum(tensors_decompressed) / self.world_size

    
    def alltoall_allgather_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        
        if is_topk_like:
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)
            return self.compressor.decompress(tensors_compressed, ctx)
        
        tensors_decompressed = []
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.world_size), metadata.chunk(self.world_size)
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))

        numel = ctx[1]
        return torch.cat(tensors_decompressed, dim=0)[:numel]


    # in-place
    def allreduce(self, tensor, async_op=True): 
        if self.world_size == 1:
            return tensor

        return self.DDPbackend.global_allreduce(tensor, async_op=async_op)