import torch
import byteps.torch as bps


class InterCommComp():
    def __init__(self, compressor, DDPbackend):
        self.compressor = compressor
        self.DDPbackend = DDPbackend
        self.local_size = bps.local_size()
        self.local_rank = bps.local_rank()
        self.worker_id = self.DDPbackend.worker_id
        self.worker_num = self.DDPbackend.worker_num


    def get_comm_root_node(self):
        return self.local_rank + (self.local_rank % self.worker_num) * self.local_size


    def gather(self, tensors_compressed):
        if self.worker_num == 1:
            return tensors_compressed

        # TODO: rotate the root_node 
        root_node = self.get_comm_root_node()
        ret = []
        for tensor in tensors_compressed:
            tensors = self.DDPbackend.gather(tensor, dst=root_node)
            ret.append(torch.stack(tensors).reshape(-1))
        return ret


    def gather_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        if is_topk_like: 
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)
            return self.compressor.decompress((tensors, metadata), ctx)

        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.worker_num), metadata.chunk(self.worker_num)
        tensors_decompressed = []
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))

        return sum(tensors_decompressed)


    def broadcast(self, tensors_compressed):
        if self.worker_num == 1:
            return tensors_compressed

        # TODO: rotate the root_node 
        root_node = self.get_comm_root_node()
        ret = []
        for tensor in tensors_compressed:
            tensors = self.DDPbackend.broadcast(tensor, src=root_node, async_op=True)
            ret.append(tensors)
        return ret


    def broadcast_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed

        if is_topk_like:
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)
            return self.compressor.decompress((tensors, metadata), ctx)
        
        tensors_decompressed = []
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.worker_num), metadata.chunk(self.worker_num)
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx, alltoall=True))
        return sum(tensors_decompressed)


    def alltoall(self, tensors_compressed, is_topk_like=True):
        if self.worker_num == 1:
            return tensors_compressed

        if not isinstance(tensors_compressed, tuple):
            tensors = self.DDPbackend.alltoall(tensors_compressed, async_op=True)
            return torch.stack(tensors).reshape(-1)

        assert(len(tensors_compressed) == 2)
        tensor, metadata = tensors_compressed
        ret = []
        if not is_topk_like:
            # signsgd-like compression algorithms
            # inter alltoall the compressed data
            data = self.DDPbackend.alltoall(tensor, async_op=True)
            # intra allgather the metadata
            metadata = self.DDPbackend.allgather(metadata, async_op=True)
        else:
            # inter alltoall the values
            data = self.DDPbackend.alltoall(tensor, async_op=True)
            # inter alltoall the indices
            metadata = self.DDPbackend.alltoall(metadata, async_op=True)
        
        ret.append(torch.stack(data).reshape(-1))
        ret.append(torch.stack(metadata).reshape(-1))
        return ret


    def alltoall_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed

        if is_topk_like:
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)
            return self.compressor.decompress((tensors, metadata), ctx, alltoall=True)

        tensors_decompressed = []
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.worker_num), metadata.chunk(self.worker_num)
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx, alltoall=True))
        return sum(tensors_decompressed)


    def allgather(self, tensors_compressed):
        if self.worker_num == 1:
            return tensors_compressed
        
        ret = []
        for tensor in tensors_compressed:
            tensors = self.DDPbackend.allgather(tensor, async_op=True)
            ret.append(torch.stack(tensors).reshape(-1))
        return ret    
        

    def allgather_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        if is_topk_like:
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)
            return self.compressor.decompress((tensors, metadata), ctx)
        
        tensors_decompressed = []
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.worker_num), metadata.chunk(self.worker_num)
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))

        return sum(tensors_decompressed)

    
    def alltoall_allgather_decompress(self, tensors_compressed, ctx, is_topk_like=True):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        if is_topk_like:
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)
            return self.compressor.decompress((tensors, metadata), ctx)
        
        tensors_decompressed = []
        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.worker_num), metadata.chunk(self.worker_num)
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))

        numel = ctx[1]
        return torch.cat(tensors_decompressed, dim=0)[:numel]