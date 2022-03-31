import torch
from mergeComp_dl.torch import Communicator
from horovod.torch import allgather, allgather_async, synchronize
import time
import horovod.torch as hvd


class PoolAllgather(Communicator):
    def __init__(self, compressor, memory):
        super().__init__(compressor, memory)
        self.world_size = hvd.size()
        self.name = "PoolAllGather"


    def async_send(self, tensors_compressed, ctx):
        if tensors_compressed is None:
            return

        handles = []
        for i, tensor_compressed in enumerate(tensors_compressed):
            handle = allgather_async(tensor_compressed, ctx[0] + str(i))
            handles.append(handle)

        return handles


    def wait_receive(self, handles, ctx):
        tensors_compressed = []
        for h in handles:
            tensor_compressed = synchronize(h)
            tensors_compressed.append(tensor_compressed.chunk(self.world_size))

        tensors_decompressed = []
        if len(tensors_compressed) == 1:
            for tensor in tensors_compressed[0]:
                tensors_decompressed.append(self.compressor.decompress([tensor], ctx))
        elif len(tensors_compressed) == 2:
            for tensor, meta in zip(tensors_compressed[0], tensors_compressed[1]):
                tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx))

        tensors_decompressed = self.memory.aggregate(tensors_decompressed)
        return tensors_decompressed
