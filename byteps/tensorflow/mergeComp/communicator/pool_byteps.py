import torch

from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import synchronize
import sys
sys.path.append("../..")
from mergeComp import Communicator

class PoolBytePS(Communicator):
    def __init__(self, compressor, memory):
        super().__init__(compressor, memory)
        self.name = "PoolBytePS"


    def async_send(self, tensors_compressed, ctx):
        return byteps_push_pull(tensors_compressed[0], average=False, name=ctx[0])


    def wait_receive(self, handle, ctx):
        output = [synchronize(handle)]
        return [self.compressor.decompress(output, ctx)]