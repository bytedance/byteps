import torch
import byteps.torch as bps
import sys
from time import time
sys.path.append("../..")
from mergeComp import Communicator


class DDPAllgatherTwolayer(Communicator):
    def __init__(self, fp16_compressor, compressor, memory, DDPbackend, profile=False):
        super().__init__(fp16_compressor, compressor, memory)
        self.intra_allgather = DDPbackend.intra_allgather
        self.inter_allgather = DDPbackend.allgather
        self.reduce_scatter = DDPbackend.intra_reduce_scatter
        self.allreduce = DDPbackend.global_allreduce
        self.threshold = 1024 * 16
        self.name = "DDPAllgatherTwolayer"
        self.local_rank = bps.local_rank()
        self.local_size = bps.local_size()
        self.world_size = bps.size()
        self.worker_id = DDPbackend.worker_id
        self.worker_num = DDPbackend.worker_num
        self.comm_stream = torch.cuda.Stream(priority=-1)
        self.handles = {}
        self.shapes = {}
        self.profile = profile
        self.compress_overhead = 0
        self.decompress_overhead = 0
        self.iteration = -1


    def is_topk_like(self):
        return self.compressor.sparsification


    def is_signsgd_like(self):
        return self.compressor.quantization


    def allgather_decompress(self, tensors_compressed, ctx, numel):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        
        # we first decompresss compressed tensors from the GPUs with the same local ID on different nodes
        # Because they are compressed from the same position range, we can sum there decompressed tensors
        # We then concatenate these sumed tensors to have the final decompressed tensor
        tensors_decompressed = []
        if self.profile:
            torch.cuda.synchronize()
            start_time = time()

        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.local_size), metadata.chunk(self.local_size)

        for tensor, meta in zip(tensors, metadata):
            if self.is_topk_like():
                tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))
            else:
                if self.worker_num == 1:
                    tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))
                else:
                    group_tensors_decompressed = []
                    group_tensor, group_metadata = tensor.chunk(self.worker_num), meta.chunk(self.worker_num)
                    for t, m in zip(group_tensor, group_metadata):
                        group_tensors_decompressed.append(self.compressor.decompress((t, m), ctx))
                    tensors_decompressed.append(sum(group_tensors_decompressed))
            
        if self.profile:
            torch.cuda.synchronize()
            self.decompress_overhead += time() - start_time
        return torch.cat(tensors_decompressed, dim=0)[:numel]


    def async_send(self, tensor, name):    
        if self.profile:
            if self.first_tensor_name is None:
                self.first_tensor_name = name
                self.iteration = 0
            elif name == self.first_tensor_name:
                self.iteration += 1
                if bps.local_rank() == 0:
                    print("Compression overhead", self.compress_overhead, ", Decompression overhead:", self.decompress_overhead, "total:", self.compress_overhead+self.decompress_overhead, flush=True)
                self.compress_overhead, self.decompress_overhead = 0, 0

        self.shapes[name] = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        # with torch.cuda.stream(self.comm_stream):
        if True:
            # we don't compress extremely small tensors
            if numel < self.threshold or numel % bps.local_size() != 0:
                self.handles[name] = self.allreduce(tensor), None, numel
                return [-1], (name,)
        
            # intra-node communication with reduce-scatter
            tensor = self.reduce_scatter(tensor)

            if self.profile:
                torch.cuda.synchronize()
                start_time = time()
            tensor = self.memory.compensate(tensor, name)
            tensor_compressed, ctx = self.compressor.compress(tensor, name)
            self.memory.update(tensor, name, self.compressor, tensor_compressed, ctx)
            
            if self.profile:
                torch.cuda.synchronize()
                self.compress_overhead += time() - start_time
            # inter-node communication with allgather
            assert(len(tensor_compressed) == 2)
            tensor, meta = tensor_compressed
            inter_tensors_compressed = [self.inter_allgather(tensor), self.inter_allgather(meta)]

            tensors, metadata = inter_tensors_compressed
            if isinstance(tensors, list):
                tensors = torch.stack(tensors).reshape(-1)
                metadata = torch.stack(metadata).reshape(-1)

            intra_tensors_compressed = [self.intra_allgather(tensors), self.intra_allgather(metadata)]
            self.handles[name] = intra_tensors_compressed, ctx, numel
            return [-1], (name,)


    def decompress_tensor(self, name):
        # with torch.cuda.stream(self.comm_stream):
        if True:
            tensors_compressed, ctx, numel = self.handles[name]
            if ctx is None:
                return tensors_compressed
            return self.allgather_decompress(tensors_compressed, ctx, numel)


    def wait_receive(self, handle, ctx):
        name = ctx[0]
        tensor = self.decompress_tensor(name)
        # torch.cuda.current_stream().wait_stream(self.comm_stream)
        return tensor.reshape(self.shapes[name])