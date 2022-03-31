import torch
import byteps.torch as bps
import sys
from time import time
sys.path.append("../..")
from mergeComp import Communicator


class DDPFP16(Communicator):
    def __init__(self, fp16_compressor, memory, DDPbackend, profile=False):
        super().__init__(fp16_compressor, fp16_compressor, memory)
        self.allreduce = DDPbackend.global_allreduce
        self.name = "DDPFP16"
        self.world_size = bps.size()
        self.comm_stream = torch.cuda.Stream(priority=-1)
        self.handles = {}
        self.shapes = {}
        self.profile = profile
        self.compress_overhead = 0
        self.decompress_overhead = 0
        self.iteration = -1


    def async_send(self, tensor, name):    
        with torch.cuda.stream(self.comm_stream):
            self.handles[name] = self.allreduce(tensor.type(torch.float16))
            return [-1], (name,)


    def wait_receive(self, handle, ctx):
        name = ctx[0]
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        return self.handles[name].type(torch.float32)