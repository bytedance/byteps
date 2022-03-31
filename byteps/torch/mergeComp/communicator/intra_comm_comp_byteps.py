import torch
import byteps.torch as bps
from byteps.torch.ops import intra_reducescatter_async, intra_allgather_async, intra_alltoall_async, intra_gather_async, intra_gather, intra_broadcast_async, intra_reduce_async
from byteps.torch.ops import synchronize
import time


class IntraCommComp():
    # TODO: memory for both intra_compressor and inter_compressor
    def __init__(self, compressor, DDPbackend):
        self.compressor = compressor
        self.DDPbackend = DDPbackend
        self.local_size = bps.local_size()


    def comm_sync(self, handles):
        if isinstance(handles, list):
            tensors = []
            for handle in handles:
                tensors.append(synchronize(handle))
            return tensors  
        else:
            return synchronize(handles)


    def reducescatter(self, tensor_compressed, name):
        return [intra_reducescatter_async(tensor_compressed, name=name)]


    def allgather(self, tensor_compressed, name, is_topk_like=True):
        handles = []
        for i, tensor in enumerate(tensor_compressed):
            if isinstance(tensor, list):
                tensor = torch.stack(tensor).reshape(-1) 
            handles.append(intra_allgather_async(tensor, name=name+str(i)))
        return handles


    def alltoall(self, tensor_compressed, name, is_topk_like=True):
        handles = []
        if not is_topk_like:
            # signsgd-like compression algorithms
            # intra alltoall the compressed data
            handles.append(intra_alltoall_async(tensor_compressed[0], name=name+"values"))
            # intra allgather the metadata
            handles.append(intra_allgather_async(tensor_compressed[1], name=name+"indices"))
        else:
            # intra alltoall the values
            handles.append(intra_alltoall_async(tensor_compressed[0], name=name+"0"))
            # intra alltoall the indices
            handles.append(intra_alltoall_async(tensor_compressed[1], name=name+"1"))
        return handles

    
    def gather(self, tensor_compressed, name, is_topk_like=True):
        handles = []
        if is_topk_like:
            for i, tensor in enumerate(tensor_compressed):
                handles.append(intra_gather_async(tensor, name=name+str(i), average=False, root=0))
        else:
            handles.append(intra_gather_async(tensor_compressed[0], name=name+"0", average=False, root=0))
            handles.append(intra_allgather_async(tensor_compressed[1], name=name+"1"))
        return handles


    def ddp_gather(self, tensor_compressed, name):
        tensors = []
        for tensor in tensor_compressed:
            tensors.append(self.DDPbackend.local_gather(tensor, async_op=True))
        return tensors

    
    def reduce(self, tensor_compressed, name):
        handles = []
        #print("[intra_alltoall] name:", name, ", rank:", self.local_rank, "value size:", len(tensor_compressed[0]), flush=True)
        for i, tensor in enumerate(tensor_compressed):
            handles.append(intra_reduce_async(tensor, average=True, name=name+str(i), root=0))
        return handles

    
    def broadcast(self, tensor_compressed, name):
        handles = []
        #print("[intra_alltoall] name:", name, ", rank:", self.local_rank, "value size:", len(tensor_compressed[0]), flush=True)
        for i, tensor in enumerate(tensor_compressed):
            if isinstance(tensor, list):
                tensor = torch.stack(tensor).reshape(-1)
            handles.append(intra_broadcast_async(tensor, name=name+str(i), root=0))
        return handles


    def ddp_broadcast(self, tensor_compressed, name):
        tensors = []
        for tensor in tensor_compressed:
            if isinstance(tensor, list):
                tensor = torch.stack(tensor).reshape(-1)
            tensors.append(self.DDPbackend.local_broadcast(tensor))
        return tensors

        
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
        return torch.cat(tensors_decompressed, dim=0)[:numel]


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

        tensors, metadata = tensors_compressed
        tensors, metadata = tensors.chunk(self.local_size), metadata.chunk(self.local_size)
        tensors_decompressed = []
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx, alltoall=True))

        return sum(tensors_decompressed)