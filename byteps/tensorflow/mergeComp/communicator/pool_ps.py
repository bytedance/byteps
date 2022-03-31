import torch
from mergeComp_dl.torch import Communicator
from horovod.torch import allgather, allgather_async, alltoall_async, synchronize, poll
import time
import copy
import threading
import horovod.torch as hvd


class ServerErrorMemory():
    def __init__(self):
        self.device = torch.cuda.current_device()
        self.error_memory = {}


    def reset_memory(self):
        self.error_memory = {}


    def get_error(self, name, size):
        if name in self.error_memory:
            return self.error_memory[name][:size]
        else:
            self.error_memory[name] = torch.zeros(size, dtype=torch.float32, device=self.device)
            return self.error_memory[name]



class PoolPS(Communicator):
    def __init__(self, compressor, memory):
        super().__init__(compressor, memory)
        self.world_size = hvd.size()
        self.rank = hvd.rank()
        self.name = "PoolPS"
        self.error_memory = ServerErrorMemory()
        self.fusion_num = memory.fusion_num
        self.memory = memory
        self.last_partition_name = self.memory.first_tensor_name
        self.reduction = self.memory.reduction


    def get_reduction(self, name):
        return self.memory.get_reduction(name)


    def get_splits(self, size):
        s = size // self.world_size
        splits = [s for i in range(self.world_size)]
        splits[-1] = size - (self.world_size-1)*s
        return torch.tensor(splits, dtype=torch.int32)


    def split_size(self, size):
        s = size // self.world_size

        if self.rank == self.world_size - 1:
            return size - (self.world_size - 1)*s
        else:
            return s


    def is_last_partition(self, name):
        return self.last_partition_name == name


    def sleep(self, time_ms):
        time.sleep(time_ms/1000)


    def ps_synchronize(self, handles, ctx):
        name, numel = ctx
        if self.is_last_partition(name):
            self.sync_receive(handles, ctx)
        else:
            self.sync_receive(handles, ctx)


    def check_handles(self, handles):
        for handle in handles:
            if not poll(handle):
                return False
        return True


    def async_send(self, tensors_compressed, ctx):
        if tensors_compressed is None:
            return

        handles = []

        """
            We use alltoall()+allgather() to implement Parameter Server.
            For quantization compression algorithms, we allgather() the corresponding scalars
            for each server to decompress the data.

        """
        name, numel = ctx

        if self.compressor.quantization and len(tensors_compressed) == 2:
            handle = alltoall_async(tensors_compressed[0], splits=self.get_splits(tensors_compressed[0].numel()), name=name)
            handles.append(handle)
            handle = allgather_async(tensors_compressed[1], name=name)
            handles.append(handle)
        else:
            for i, tensor_compressed in enumerate(tensors_compressed):
                handle = alltoall_async(tensor_compressed, name + str(i))
                handles.append(handle)

        #self.thread = threading.Thread(target=self.ps_synchronize, args=(handles, ctx))
        #self.thread.start()
        self.ps_synchronize(handles, ctx)

        return handles


    def sync_receive(self, handles, ctx):
        name, numel = ctx

        # receive data from all workers
        tensors_compressed = []
        for h in handles:
            tensor_compressed = synchronize(h)
            tensors_compressed.append(tensor_compressed.chunk(self.world_size))

        name, numel = ctx
        ctx_ps = name, self.split_size(numel)

        tensors_decompressed = []

        # decompress the data
        if len(handles) == 1:
            for tensor in tensors_compressed[0]:
                tensors_decompressed.append(self.compressor.decompress(tensor, ctx_ps))
        elif len(handles) == 2:
            for tensor, meta in zip(tensors_compressed[0], tensors_compressed[1]):
                tensors_decompressed.append(self.compressor.decompress((tensor, meta), ctx_ps))

        # reduce the decompressed data
        tensors_reduced = sum(tensors_decompressed)

        # compensate the reduced data before compression
        error_memory = self.error_memory.get_error(name, len(tensors_reduced))
        tensors_reduced.add_(error_memory)

        # compress the reduced data again
        tensor_compressed, ctx_allgather = self.compressor.compress(tensors_reduced, name, None, server=True)
        error_memory.set_(tensors_reduced - self.compressor.decompress(tensor_compressed, ctx_allgather, server=True))

        reduction = self.get_reduction(name)
        pre_offset = 0
        offset = 0
        if len(tensor_compressed) == 1:
            # allgather the compressed tensor
            worker_tensors_compressed = allgather(tensor_compressed, ctx[0])
            # decompress the compressed tensors
            for tensor in worker_tensors_compressed[0].chunk(self.world_size):
                tensor_decompressed = self.compressor.decompress(tensor, ctx_allgather)
                offset += tensor_decompressed.numel()
                reduction[pre_offset:offset].set_(tensor_decompressed)
                pre_offset = offset
        elif len(tensor_compressed) == 2:
            # allgather the compressed tensor and meta
            worker_tensors_compressed = allgather(tensor_compressed[0], ctx[0] +'0').chunk(self.world_size)
            metas = allgather(tensor_compressed[1], ctx[0]+'1').chunk(self.world_size)
            # decompress the compressed tensors
            for tensor_compressed, meta in zip(worker_tensors_compressed, metas):
                tensor_decompressed = self.compressor.decompress((tensor_compressed, meta), ctx_allgather)
                offset += tensor_decompressed.numel()
                reduction[pre_offset:offset].set_(tensor_decompressed)
                pre_offset = offset


    def wait_receive(self, handles, ctx):
        return None
