from abc import ABC, abstractmethod
import time


class Memory(ABC):
    def initialize(self, named_parameters):
        pass

    def pool_compensate(self, tensor, name):
        pass

    def pool_update(self, tensor, name, compressor, tensor_compressed, ctx, advance_reduce=False):
        """Update the residuals."""
        pass

    def pool_reduce(self, compressed_tensor):
        pass


class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name, ctx=None, server=False):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx, server=False):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)

    def clean(self):
        pass


class Communicator(ABC):
    def __init__(self, compressor, memory):
        self.compressor = compressor
        self.memory = memory


    def send_step(self, tensor):
        name = tensor.name
        tensor, meta = self.memory.pool_compensate(tensor, name)
        if tensor is not None:
            #print("[send_step]", name)
            tensor_compressed, ctx = self.compressor.compress(tensor, name, meta)
            self.memory.pool_update(tensor, name, self.compressor, tensor_compressed, ctx)
            return tensor_compressed, ctx

        return None, None


    def receive_step(self, tensors, ctx):
        name = ctx[0]
        #print("[receive step]", name)
        decompressed_tensors = self.compressor.decompress(tensors, ctx)
        if tensors is not None:
            self.memory.reduce(decompressed_tensors, name)
        return self.memory.pool_step(name)