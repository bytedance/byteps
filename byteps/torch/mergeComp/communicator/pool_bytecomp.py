import torch
from byteps.torch.ops import cpu_compress_async
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import synchronize
import byteps.torch as bps
import sys, os
import time
import copy
from threading import Thread
from queue import Queue, Empty
from .intra_comm_comp import IntraCommComp
from .inter_comm_comp import InterCommComp
from .global_comm_comp import GlobalCommComp
sys.path.append("../..")
from mergeComp import Communicator


"""
@comm_type: Communication type
0: FP32
1: FP16
2: intra-node FP16 + inter-node compression + gather + broadcast
3: intra-node FP16 + inter-node compression + alltoall + allgather
4: intra-node FP16 + inter-node compression + allgather
5: intra-node FP16 + CPU compression
6: intra-node compression + intra-node alltoall + inter-node compression + gather + broadcast + intra-node allgather
7: intra-node compression + intra-node alltoall + inter-node compression + alltoall + allgather + intra-node allgather
8: intra-node compression + intra-node gather + inter-node compression + allgather + intra-node broadcast
9: intra-node compression + intra-node alltoall + inter-node compression + allgather + intra-node allgather
10: allgather
11: alltoall
"""
class ByteComp(Communicator):
    def __init__(self, fp16_compressor, compressor, memory, DDPbackend, scheduler):
        super().__init__(fp16_compressor, compressor, memory)
        self.scheduler = scheduler
        self.init_error_feedback_memory(memory)
        self.name = "PoolByteComp"
        self.device = torch.cuda.current_device()
        self.local_size = bps.local_size()
        self.local_rank = bps.local_rank()
        self.world_size = bps.size()
        assert(self.world_size % self.local_size == 0)
        self.worker_num = self.world_size // self.local_size
        self.intra_comm_comp = IntraCommComp(compressor, DDPbackend)
        self.inter_comm_comp = InterCommComp(compressor, DDPbackend)
        self.global_comm_comp = GlobalCommComp(compressor, DDPbackend)

        self.handles = {}
        self.intra_ctx = {}
        self.shapes = {}
        self.tensors_comm_type = {}
        self.comm_stream = torch.cuda.Stream(priority=-1)


    def init_error_feedback_memory(self, memory):
        self.memories = []
        """
        there are three positions required error-feedback memory.
        0: before intra-node communication
        1: after intra-node communication
        2: during inter-node communication 
        """
        memory_num = 3
        for _ in range(memory_num):
            self.memories.append(copy.deepcopy(memory))


    def is_topk_like(self):
        return self.compressor.sparsification


    def is_signsgd_like(self):
        return self.compressor.quantization


    def get_FP32_intra_comm(self, tensor):
        numel = tensor.numel()
        return self.intra_comm_comp.reducescatter(tensor), numel


    def get_FP16_intra_comm(self, tensor, name):
        tensor_fp16, ctx = self.fp16_compressor.compress(tensor, name)
        tensor = self.intra_comm_comp.reducescatter(tensor_fp16[0])
        return tensor, ctx


    def compress(self, tensor, name, memory_id=1, alltoall_nodes=1, unit_size=8):
        memory = self.memories[memory_id]
        tensor = memory.compensate(tensor, name)
        if self.is_topk_like():
            tensor_compressed, ctx = self.compressor.compress(tensor, name, alltoall_nodes=alltoall_nodes)
        else:
            tensor_compressed, ctx = self.compressor.compress(tensor, name, signsgd_unit_size=unit_size, alltoall_nodes=alltoall_nodes)
        memory.update(tensor, name, self.compressor, tensor_compressed, ctx)
        return tensor_compressed, ctx
            

    def comm_type_2(self, tensor, name):
        # inter-node comm: gather + broadcast
        with torch.cuda.stream(self.comm_stream):
            tensor, intra_ctx = self.get_FP16_intra_comm(tensor, name)
            self.intra_ctx[name] = intra_ctx

            tensor_compressed, ctx = self.compress(tensor, name, memory_id=1)
            tensors_compressed = self.inter_comm_comp.gather(tensor_compressed)
            root_node = self.inter_comm_comp.get_comm_root_node()
            if bps.rank == root_node:
                # only the root GPU needs to gather the message
                tensor_decompressed = self.inter_comm_comp.gather_decompress(tensors_compressed, ctx, self.is_topk_like())
                tensor_compressed, _ = self.compress(tensor_decompressed, name, memory_id=2)
            
            # broadcast the compressed message to all GPUs in the same inter-node communication group
            tensor_compressed_broadcast = self.inter_comm_comp.broadcast(tensor_compressed)
            # decompress the messages and allgather among all GPUs in the same machine
            intra_comm_tensor = self.inter_comm_comp.broadcast_decompress(tensor_compressed_broadcast, ctx)
            if self.is_signsgd_like():
                # convert FP32 to FP16 for intra-node communication
                intra_comm_tensor = intra_comm_tensor.type(torch.float16)
            self.handles[name] = self.intra_comm_comp.allgather([intra_comm_tensor])


    def comm_type_3(self, tensor, name):
        # inter-node comm: alltoall + allgather
        with torch.cuda.stream(self.comm_stream):
            tensor, intra_ctx = self.get_FP16_intra_comm(tensor, name)
            self.intra_ctx[name] = intra_ctx

            tensor_compressed, ctx = self.compress(tensor, name, memory_id=1, alltoall_nodes=self.worker_num)
            tensors_compressed = self.inter_comm_comp.alltoall(tensor_compressed, self.is_topk_like())
            tensor_decompressed = self.inter_comm_comp.alltoall_decompress(tensors_compressed, ctx, self.is_topk_like())
            tensor_compressed, _ = self.compress(tensor_decompressed, name, memory_id=2)
            # allgather the compressed message in the same inter-node communication group
            tensor_compressed_allgather = self.inter_comm_comp.allgather(tensor_compressed)
            # decompress the messages and allgather among all GPUs in the same machine
            intra_comm_tensor = self.inter_comm_comp.alltoall_allgather_decompress(tensor_compressed_allgather, ctx, self.is_topk_like())
            if self.is_signsgd_like():
                # convert FP32 to FP16 for intra-node communication
                intra_comm_tensor = intra_comm_tensor.type(torch.float16)
            self.handles[name] = self.intra_comm_comp.allgather([intra_comm_tensor])


    def comm_type_4(self, tensor, name):
        # inter-node comm: allgather
        with torch.cuda.stream(self.comm_stream):
            tensor, intra_ctx = self.get_FP16_intra_comm(tensor, name)
            self.intra_ctx[name] = intra_ctx

            tensor_compressed, ctx = self.compress(tensor, name, memory_id=1)
            tensors = self.inter_comm_comp.allgather(tensor_compressed)
            intra_comm_tensor = self.inter_comm_comp.allgather_decompress(tensors, ctx, self.is_topk_like())
            if self.is_signsgd_like():
                # convert FP32 to FP16 for intra-node communication
                intra_comm_tensor = intra_comm_tensor.type(torch.float16)
            self.handles[name] = self.intra_comm_comp.allgather([intra_comm_tensor])


    def comm_type_6(self, tensor, name):
        # intra-node compression + intra-node alltoall + inter-node compression + gather + broadcast + intra-node allgather
        with torch.cuda.stream(self.comm_stream):
            tensor_compressed, intra_ctx = self.compress(tensor, name, memory_id=0, alltoall_nodes=self.local_size)
            # after compress() with alltoall flags, the compressed message can be evenly partitioned.
            self.intra_ctx[name] = intra_ctx

            tensor_alltoall = self.intra_comm_comp.alltoall(tensor_compressed, self.is_topk_like())
            inter_tensor = self.intra_comm_comp.alltoall_decompress(tensor_alltoall, intra_ctx, self.is_topk_like())
            tensor_compressed, ctx = self.compress(inter_tensor, name, memory_id=1)  

            tensors_compressed = self.inter_comm_comp.gather(tensor_compressed)
            root_node = self.inter_comm_comp.get_comm_root_node()
            if bps.rank == root_node:
                # only the root GPU needs to gather the message
                tensor_decompressed = self.inter_comm_comp.gather_decompress(tensors_compressed, ctx, self.is_topk_like())
                tensor_compressed, _ = self.compress(tensor_decompressed, name, memory_id=2)
            # broadcast the compressed message to all GPUs in the same inter-node communication group
            tensor_compressed_broadcast = self.inter_comm_comp.broadcast(tensor_compressed)
            tensor_compressed_broadcast = tensor_compressed
            self.handles[name] = self.intra_comm_comp.allgather(tensor_compressed_broadcast)


    def comm_type_7(self, tensor, name):
        # intra-node compression + intra-node alltoall + inter-node compression + alltoall + allgather + intra-node allgather
        with torch.cuda.stream(self.comm_stream):
            tensor_compressed, intra_ctx = self.compress(tensor, name, memory_id=0, alltoall_nodes=self.local_size)
            # after compress() with alltoall flags, the compressed message can be evenly partitioned.
            self.intra_ctx[name] = intra_ctx

            tensor_alltoall = self.intra_comm_comp.alltoall(tensor_compressed, self.is_topk_like())
            inter_tensor = self.intra_comm_comp.alltoall_decompress(tensor_alltoall, intra_ctx, self.is_topk_like())
            tensor_compressed, ctx = self.compress(inter_tensor, name, memory_id=1)  

            tensors_compressed = self.inter_comm_comp.alltoall(tensor_compressed, self.is_topk_like())
            tensor_decompressed = self.inter_comm_comp.alltoall_decompress(tensors_compressed, ctx, self.is_topk_like())
            tensor_compressed, _ = self.compress(tensor_decompressed, name, memory_id=2)
            # allgather the compressed message in the same inter-node communication group

            tensor_compressed_allgather = self.inter_comm_comp.allgather(tensor_compressed)
            # if self.is_signsgd_like():
            #     tensor_compressed_allgather = tensor_compressed_allgather[0].type(torch.float16), tensor_compressed_allgather[1]
            
            self.handles[name] = self.intra_comm_comp.allgather(tensor_compressed_allgather)
            
 
    def comm_type_8(self, tensor, name):
        # intra-node compression + intra-node gather + inter-node compression + allgather + intra-node broadcast
        with torch.cuda.stream(self.comm_stream):
            tensor_compressed, intra_ctx = self.compress(tensor, name, memory_id=0)
            self.intra_ctx[name] = intra_ctx

            tensor_gather = self.intra_comm_comp.gather(tensor_compressed)
            if self.local_rank == 0:
                tensor = self.intra_comm_comp.gather_decompress(tensor_gather, intra_ctx, self.is_topk_like())
                tensor_compressed, _ = self.compress(tensor, name, memory_id=1)
                output = self.inter_comm_comp.allgather(tensor_compressed)
                self.handles[name] = self.intra_comm_comp.broadcast(output)
            else:
                # assume all workers have the same data size
                output = []
                for i in range(len(tensor_gather)):
                    output_shape = list(tensor_gather[i].shape)
                    output_shape[0] = output_shape[0] * self.worker_num
                    output.append(torch.empty(output_shape, dtype=tensor_gather[i].dtype, device=tensor_gather[i].device))
                self.handles[name] = self.intra_comm_comp.broadcast(output)


    def comm_type_9(self, tensor, name):
        # intra-node compression + intra-node alltoall + inter-node compression + allgather + intra-node allgather
        with torch.cuda.stream(self.comm_stream):
            tensor_compressed, intra_ctx = self.compress(tensor, name, memory_id=0, alltoall_nodes=self.local_size)
            self.intra_ctx[name] = intra_ctx

            tensor_alltoall = self.intra_comm_comp.alltoall(tensor_compressed, self.is_topk_like())
            inter_tensor = self.intra_comm_comp.alltoall_decompress(tensor_alltoall, intra_ctx, self.is_topk_like())
            tensor_compressed, ctx = self.compress(inter_tensor, name, memory_id=1)  

            tensors_compressed = self.inter_comm_comp.allgather(tensor_compressed)
            self.handles[name] = self.intra_comm_comp.allgather(tensors_compressed)


    def comm_type_10(self, tensor, name):
        # global allgather
        with torch.cuda.stream(self.comm_stream):
            tensor_compressed, intra_ctx = self.compress(tensor, name, memory_id=0)
            self.intra_ctx[name] = intra_ctx
            self.handles[name] = self.global_comm_comp.allgather(tensor_compressed)

    
    def comm_type_11(self, tensor, name):
        # global alltoall + allgather
        with torch.cuda.stream(self.comm_stream):
            tensor_compressed, intra_ctx = self.compress(tensor, name, memory_id=0, alltoall_nodes=self.world_size)
            self.intra_ctx[name] = intra_ctx
            alltoall_tensors = self.global_comm_comp.alltoall(tensor_compressed)
            decompressed_tensor = self.global_comm_comp.alltoall_decompress(alltoall_tensors, intra_ctx, self.is_topk_like())
            allgather_tensor_compressed, ctx = self.compress(decompressed_tensor, name, memory_id=1)
            self.handles[name] = self.global_comm_comp.allgather(allgather_tensor_compressed)

    
    def comm_type_12(self, tensor, name):
        # intra-node compression + intra-node alltoall + inter-node compression + allgather + decompress + compress intra-node allgather
        with torch.cuda.stream(self.comm_stream):
            tensor_compressed, intra_ctx = self.compress(tensor, name, memory_id=0, alltoall_nodes=self.local_size)
            self.intra_ctx[name] = intra_ctx

            tensor_alltoall = self.intra_comm_comp.alltoall(tensor_compressed, self.is_topk_like())
            inter_tensor = self.intra_comm_comp.alltoall_decompress(tensor_alltoall, intra_ctx, self.is_topk_like())
            tensor_compressed, ctx = self.compress(inter_tensor, name, memory_id=1)  

            tensors_compressed = self.inter_comm_comp.allgather(tensor_compressed)
            tensor_decompressed = self.inter_comm_comp.allgather_decompress(tensors_compressed, ctx, self.is_topk_like())
            intra_tensor_compressed, ctx = tensor_compressed, ctx = self.compress(tensor_decompressed, name, memory_id=2)  
            self.handles[name] = self.intra_comm_comp.allgather(intra_tensor_compressed)


    def comm_synchronize(self, tensor, name, comm_type):
        if comm_type == 2:
            self.comm_type_2(tensor, name)
        elif comm_type == 3:
            self.comm_type_3(tensor, name)
        elif comm_type == 4:
            self.comm_type_4(tensor, name)
        elif comm_type == 6:
            self.comm_type_6(tensor, name)
        elif comm_type == 7:
            self.comm_type_7(tensor, name)
        elif comm_type == 8:
            self.comm_type_8(tensor, name)
        elif comm_type == 9:
            self.comm_type_9(tensor, name)
        elif comm_type == 10:
            self.comm_type_10(tensor, name)
        elif comm_type == 11:
            self.comm_type_11(tensor, name)
        elif comm_type == 12:
            self.comm_type_12(tensor, name)


    def get_comm_type(self, name, size=2**30):
        return self.scheduler.get_comm_type(name, size)


    def comm_comp(self, tensor, name):
        comm_type = self.tensors_comm_type[name]
        handle, ctx = None, (name, None)

        # if comm_type == 0:
        #     self.handles[name], self.intra_ctx[name] = byteps_push_pull(tensor, average=True, name="FP32."+name), ctx
        #     return 
        # if comm_type == 1:
        #     # FP16
        #     tensor_fp16, ctx = tensor.type(torch.float16), None
        #     self.handles[name], self.intra_ctx[name] = byteps_push_pull(tensor_fp16, average=True, name="FP16."+name), ctx
        #     return 
        # if comm_type == 5:
        #     # cpu compression: intra-node communication with FP16 and compress the tensors with CPU for inter-node comm
        #     tensor_fp16, ctx = tensor.type(torch.float16), None
        #     handle = cpu_compress_async(tensor_fp16, average=True, name=name)
        #     self.handles[name], self.intra_ctx[name] = handle, ctx
        #     return
        
        with torch.cuda.stream(self.comm_stream):
            if comm_type == 0:
                # FP32
                self.handles[name], self.intra_ctx[name] = self.global_comm_comp.allreduce(tensor), ctx
                return
            elif comm_type == 1:
                # FP16
                tensor_fp16, ctx = tensor.type(torch.float16), None
                self.handles[name], self.intra_ctx[name] = self.global_comm_comp.allreduce(tensor_fp16), ctx
                return 
            elif comm_type == 5:
                # cpu compression: intra-node communication with FP16 and compress the tensors with CPU for inter-node comm
                tensor_fp16, ctx = self.fp16_compressor.compress(tensor, name)
                handle = cpu_compress_async(tensor_fp16[0], average=True, name=name)
                self.handles[name], self.intra_ctx[name] = handle, ctx
                return

        self.comm_synchronize(tensor, name, comm_type)


    def async_send(self, tensor, name):  
        self.shapes[name] = tensor.size()
        if name not in self.tensors_comm_type:
            self.tensors_comm_type[name] = self.get_comm_type(name, tensor.numel())
        comm_type = self.tensors_comm_type[name]
        
        if comm_type not in (0, 1):
            tensor = tensor.flatten() 

        self.comm_comp(tensor, name)
        return [-1], (name,)


    def decompress_tensor(self, name, comm_type):
        # if comm_type == 0:
        #     # FP32
        #     return synchronize(self.handles[name])
        # if comm_type == 1:
        #     # FP16
        #     tensor = synchronize(self.handles[name])
        #     return tensor.type(torch.float32)
        # if comm_type == 5:
        #     # CPU compression
        #     tensor = synchronize(self.handles[name])
        #     return tensor.type(torch.float32)

        with torch.cuda.stream(self.comm_stream):
            tensor = self.handles[name]
            if comm_type == 0:
                # FP32
                return tensor
            elif comm_type == 1:
                # FP16
                return tensor.type(torch.float32)
            elif comm_type == 5:
                # CPU compression
                tensor = synchronize(tensor)
                ctx = self.intra_ctx[name]
                return tensor.type(torch.float32)

            if comm_type in (2, 3, 4):
                # FP16
                # return tensor[0]
                ctx = self.intra_ctx[name]
                return tensor[0].type(torch.float32)
            elif comm_type in (6, 7):
                # intra-node compression + intra-node alltoall + inter-node compression
                ctx = self.intra_ctx[name]
                return self.intra_comm_comp.allgather_decompress(tensor, ctx)
            elif comm_type == 8:
                ctx = self.intra_ctx[name]
                return self.intra_comm_comp.broadcast_decompress(tensor, ctx, self.is_topk_like(), self.worker_num)
            elif comm_type == 9:
                ctx = self.intra_ctx[name]
                return self.intra_comm_comp.allgather_decompress(tensor, ctx)
            elif comm_type == 10:
                ctx = self.intra_ctx[name]
                return self.global_comm_comp.allgather_decompress(tensor, ctx, self.is_topk_like())
            elif comm_type == 11:
                ctx = self.intra_ctx[name]
                return self.global_comm_comp.alltoall_allgather_decompress(tensor, ctx, self.is_topk_like())
            elif comm_type == 12:
                ctx = self.intra_ctx[name]
                return self.intra_comm_comp.allgather_decompress(tensor, ctx)


    def wait_receive(self, handle, ctx):
        name = ctx[0]
        comm_type = self.tensors_comm_type[name]
        
        tensor = self.decompress_tensor(name, comm_type)
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        if comm_type not in (0, 1):
            tensor = tensor.reshape(self.shapes[name])
        return tensor