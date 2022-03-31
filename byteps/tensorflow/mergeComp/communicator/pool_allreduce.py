import torch
from mergeComp_dl.torch import Communicator
from horovod.torch import allreduce_async, synchronize
from horovod.torch.mpi_ops import Average


class PoolAllreduce(Communicator):
    def __init__(self, compressor, memory):
        super().__init__(compressor, memory)
        self.name = "PoolAllReduce"


    def async_send(self, tensors_compressed, ctx):
        # assert only one tensor in tensors_compressed for allreduce
        return allreduce_async(tensors_compressed[0], name=ctx[0], op=Average)


    def wait_receive(self, handle, ctx):
        output = [synchronize(handle)]
        return [self.compressor.decompress(output, ctx)]

