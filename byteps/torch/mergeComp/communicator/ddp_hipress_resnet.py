import torch
import byteps.torch as bps
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import synchronize
import sys
from time import time
sys.path.append("../..")
from mergeComp import Communicator


class DDPHiPressResNet(Communicator):
    def __init__(self, fp16_compressor, compressor, memory, DDPbackend, threshold=1024 * 128 * 8, profile=False):
        super().__init__(fp16_compressor, compressor, memory)
        self.intra_allgather = DDPbackend.intra_allgather
        self.inter_allgather = DDPbackend.allgather
        self.reduce_scatter = DDPbackend.intra_reduce_scatter
        self.allreduce = byteps_push_pull
        self.threshold = threshold
        self.name = "DDPHiPressResNet"
        self.local_rank = bps.local_rank()
        self.local_size = bps.local_size()
        self.world_size = bps.size()
        self.worker_id = DDPbackend.worker_id
        self.worker_num = DDPbackend.worker_num
        self.comm_stream = torch.cuda.Stream(priority=-1)
        self.handles = {}
        self.shapes = {}
        self.tensor_sizes = {}
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
        if self.is_topk_like():
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
        self.tensor_sizes[name] = numel

        with torch.cuda.stream(self.comm_stream):
            if numel < self.threshold or numel % bps.local_size() != 0:
                self.handles[name] = self.allreduce(tensor, average=True, name=name)
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

            tensors_decompressed = self.allgather_decompress((tensors, metadata), ctx)
            self.handles[name] = self.intra_allgather(tensors_decompressed)
            return [-1], (name,)


    def wait_receive(self, handle, ctx):
        name = ctx[0]
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        numel = self.tensor_sizes[name]
        if numel < self.threshold or numel % bps.local_size() != 0:
            tensor = synchronize(self.handles[name])
        else:
            tensor = self.handles[name]
        if isinstance(tensor, list):
            tensor = torch.stack(tensor).reshape(-1)
        return tensor.reshape(self.shapes[name])