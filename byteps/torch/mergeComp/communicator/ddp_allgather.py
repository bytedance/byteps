import torch
import byteps.torch as bps
import sys
from time import time
sys.path.append("../..")
from mergeComp import Communicator


class DDPAllgather(Communicator):
    def __init__(self, fp16_compressor, compressor, memory, DDPbackend, profile=False):
        super().__init__(fp16_compressor, compressor, memory)
        self.allgather = DDPbackend.global_allgather
        self.allreduce = DDPbackend.global_allreduce
        self.name = "DDPAllgather"
        self.world_size = bps.size()
        self.comm_stream = torch.cuda.Stream(priority=0)
        self.handles = {}
        self.shapes = {}
        self.threshold = 1024 * 16
        self.profile = profile
        self.compress_overhead = 0
        self.decompress_overhead = 0
        self.iteration = -1


    def is_topk_like(self):
        return self.compressor.sparsification

    def is_signsgd_like(self):
        return self.compressor.quantization


    def allgather_decompress(self, tensors_compressed, ctx):
        assert(len(tensors_compressed) == 2)
        tensors, metadata = tensors_compressed
        tensors_decompressed = []
        if self.profile:
            torch.cuda.synchronize()
            start_time = time()

        if self.is_topk_like():
            tensors, metadata = torch.stack(tensors).reshape(-1), torch.stack(metadata).reshape(-1)
            return self.compressor.decompress((tensors, metadata), ctx)

        if not isinstance(tensors, list):
            tensors, metadata = tensors.chunk(self.world_size), metadata.chunk(self.world_size)
        for tensor, meta in zip(tensors, metadata):
            tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))
        
        if self.profile:
            torch.cuda.synchronize()
            self.decompress_overhead += time() - start_time

        numel = ctx[1]
        tensor_decompressed = sum(tensors_decompressed)
        return tensor_decompressed[:numel]


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

        with torch.cuda.stream(self.comm_stream):
            if numel < self.threshold:
                self.handles[name] = self.allreduce(tensor)
                return [-1], (name,)

            if self.profile:
                torch.cuda.synchronize()
                start_time = time()
            tensor = self.memory.compensate(tensor, name)
            tensor_compressed, ctx = self.compressor.compress(tensor, name)
            self.memory.update(tensor, name, self.compressor, tensor_compressed, ctx)

            if self.profile:
                torch.cuda.synchronize()
                self.compress_overhead += time() - start_time

            tensors_compressed = []
            assert(len(tensor_compressed) == 2)
            for tensor in tensor_compressed:
                tensors_compressed.append(self.allgather(tensor))

            tensor_decompressed = self.allgather_decompress(tensors_compressed, ctx)
            self.handles[name] = tensor_decompressed
            return [-1], (name,)


    def wait_receive(self, handle, ctx):
        name = ctx[0]
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        tensor = self.handles[name]
        return tensor.reshape(self.shapes[name])