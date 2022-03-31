import torch
import torch.distributed as dist
from launch_bps import launch_bps
import byteps.torch as bps


class DDPBackend():
    def __init__(self):
        self.local_rank = bps.local_rank()
        device = torch.device("cuda:{}".format(self.local_rank))
        torch.cuda.set_device(device)
        self.set_groups()


    @staticmethod
    def init(local_rank, backend="nccl"):
        dist.init_process_group(backend=backend, init_method='env://')
        launch_bps(local_rank)


    def set_groups(self):
        self.local_size = bps.local_size()
        self.local_rank = bps.local_rank()
        self.world_size = bps.size()
        self.rank = bps.rank()
        # TODO: handle the case if workers have different number of GPUs
        assert(self.world_size % self.local_size == 0)
        self.worker_id = self.rank // self.local_size
        self.worker_num = self.world_size // self.local_size

        self.comm_ranks_groups = []
        for i in range(self.local_size):
            ranks_group = [rank for rank in range(i, self.world_size, self.local_size)]
            assert(len(ranks_group) == self.worker_num)
            self.comm_ranks_groups.append(ranks_group)
        self.comm_groups = [dist.new_group(group) for group in self.comm_ranks_groups]

        # the communication group for GPUs in the same machine
        self.local_comm_ranks_groups = []
        for i in range(self.worker_num):
            base_rank = i * self.local_size
            local_ranks_group = [rank+base_rank for rank in range(0, self.local_size)]
            self.local_comm_ranks_groups.append(local_ranks_group)
        self.local_comm_groups = [dist.new_group(group) for group in self.local_comm_ranks_groups]


    def get_comm_group(self):
        return self.comm_groups[self.local_rank], self.worker_num


    def get_local_comm_group(self):
        return self.local_comm_groups[self.worker_id], self.local_size


    def get_ranks_group(self):
        return self.comm_ranks_groups[self.local_rank]


    def get_local_ranks_group(self):
        return self.local_comm_ranks_groups[self.worker_id]

    # in-place
    def broadcast(self, tensor, src, async_op=True):
        """
        @src: the global rank of the GPU that broadcasts the message
        """
        if self.worker_num == 1:
            return [tensor]

        comm_group, _ = self.get_comm_group()
        if async_op:
            handle = dist.broadcast(tensor, src=src, group=comm_group, async_op=async_op)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.broadcast(tensor, src=src, group=comm_group)
        return tensor


    def gloo_gather(self, tensor, dst, async_op=True):
        """
        @dst: the global rank of the GPU that gathers the message
        """
        if self.worker_num == 1:
            return [tensor]

        comm_group, group_size = self.get_comm_group()
        if dst == self.rank:
            ret = [torch.empty_like(tensor) for _ in range(group_size)]
            if async_op:
                handle = dist.gather(tensor, dst=dst, gather_list=ret, group=comm_group, async_op=True)
                handle.wait()
            else:
                dist.gather(tensor, dst=dst, gather_list=ret, group=comm_group)
            return ret
        else:
            if async_op:
                handle = dist.gather(tensor, dst=dst, group=comm_group, async_op=True)
                handle.wait()
            else:
                dist.gather(tensor, dst=dst, group=comm_group)
            return tensor


    def my_gather(self, tensor, dst):
        req = []
        comm_group, group_size = self.get_comm_group()
        ranks_group = self.get_ranks_group()
        if dst == self.rank:
            ret = [torch.empty_like(tensor) for _ in range(group_size)]
            idx = 0
            for rank in ranks_group:
                if rank != self.rank:
                    req.append(dist.irecv(ret[idx], group=comm_group, src=rank))
                else:
                    ret[idx] = tensor
                idx += 1
        else:
            req.append(dist.isend(tensor=tensor, group=comm_group, dst=dst))
            ret = [tensor]
        
        for r in req:
            r.wait()
        return ret


    def gather(self, tensor, dst, async_op=True):
        return self.my_gather(tensor, dst)

    
    def intra_gather(self, tensor, dst, async_op=True):
        req = []
        comm_group, group_size = self.get_local_comm_group()
        ranks_group = self.get_local_ranks_group()
        if dst == self.local_rank:
            ret = [torch.empty_like(tensor) for _ in range(group_size)]
            idx = 0
            for rank in ranks_group:
                if rank != self.rank:
                    req.append(dist.irecv(ret[idx], group=comm_group, src=rank))
                else:
                    ret[idx] = tensor
                idx += 1
        else:
            gloabl_dst = self.local_size * self.worker_id + dst
            req.append(dist.isend(tensor=tensor, group=comm_group, dst=gloabl_dst))
            ret = [tensor]
        
        for r in req:
            r.wait()
        return ret


    def allgather(self, tensor, async_op=True):
        if self.worker_num == 1:
            return [tensor]
            
        comm_group, group_size = self.get_comm_group()
        ret = [torch.empty_like(tensor) for _ in range(group_size)]
        if async_op:
            handle = dist.all_gather(ret, tensor, group=comm_group, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_gather(ret, tensor, group=comm_group)
        return ret


    # in-place
    def allreduce(self, tensor, async_op=True):
        if self.worker_num == 1:
            return [tensor]
            
        comm_group, _ = self.get_comm_group()
        if async_op:
            handle = dist.all_reduce(tensor, group=comm_group, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_reduce(tensor, group=comm_group)
        return tensor


    def gloo_alltoall(self, tensor, async_op=True):
        if self.worker_num == 1:
            return [tensor]
            
        comm_group, group_size = self.get_comm_group()
        comm_ranks_group = self.get_ranks_group()
        if isinstance(tensor, list): 
            assert(len(tensor) == group_size)
            output_tensor = [torch.empty_like(tensor[self.worker_id]) for _ in range(group_size)]
        else:
            numel = tensor.numel()
            elem_per_gpu = numel // group_size
            input_splits = [elem_per_gpu for _ in range(group_size)]
            input_splits[-1] = numel - elem_per_gpu * (group_size - 1)
            tensor = tensor.split(input_splits)
            output_tensor = [torch.empty_like(tensor[self.worker_id]) for _ in range(group_size)]

        for i in range(group_size):
            dst = comm_ranks_group[i]
            if async_op:
                if self.rank == dst:
                    handle = dist.gather(tensor[i], dst=dst, gather_list=output_tensor, group=comm_group, async_op=True)
                else:
                    handle = dist.gather(tensor[i], dst=dst, group=comm_group, async_op=True)
                handle.wait()
            else:
                if self.rank == dst:
                    handle = dist.gather(tensor[i], dst=dst, gather_list=output_tensor, group=comm_group)
                else:
                    handle = dist.gather(tensor[i], dst=dst, group=comm_group)

        return output_tensor


    def my_alltoall(self, tensor): 
        comm_group, group_size = self.get_comm_group()
        comm_ranks_group = self.get_ranks_group()
        if isinstance(tensor, list): 
            assert(len(tensor) == group_size)
            ret = [torch.empty_like(tensor[self.worker_id]) for _ in range(group_size)]
        else:
            numel = tensor.numel()
            elem_per_gpu = numel // group_size
            input_splits = [elem_per_gpu for _ in range(group_size)]
            input_splits[-1] = numel - elem_per_gpu * (group_size - 1)
            tensor = tensor.split(input_splits)
            ret = [torch.empty_like(tensor[self.worker_id]) for _ in range(group_size)]

        req = []
        idx = 0
        for rank in comm_ranks_group:
            if rank != self.rank:
                req.append(dist.isend(tensor=tensor[idx], group=comm_group, dst=rank))
                req.append(dist.irecv(ret[idx], group=comm_group, src=rank))
            else:
                ret[idx] = tensor[idx]
            idx += 1
        
        for r in req:
            r.wait()
        return ret


    def alltoall(self, tensor, async_op=True):
        return self.my_alltoall(tensor)


    def intra_alltoall(self, tensor, async_op=True):
        comm_group, group_size = self.get_local_comm_group()
        comm_ranks_group = self.get_local_ranks_group()
        if isinstance(tensor, list): 
            assert(len(tensor) == group_size)
            ret = [torch.empty_like(tensor[self.local_rank]) for _ in range(group_size)]
        else:
            numel = tensor.numel()
            elem_per_gpu = numel // group_size
            input_splits = [elem_per_gpu for _ in range(group_size)]
            input_splits[-1] = numel - elem_per_gpu * (group_size - 1)
            tensor = tensor.split(input_splits)
            ret = [torch.empty_like(tensor[self.local_rank]) for _ in range(group_size)]

        req = []
        idx = 0
        for rank in comm_ranks_group:
            if rank != self.rank:
                req.append(dist.isend(tensor=tensor[idx], group=comm_group, dst=rank))
                req.append(dist.irecv(ret[idx], group=comm_group, src=rank))
            else:
                ret[idx] = tensor[idx]
            idx += 1
        
        for r in req:
            r.wait()
        return ret


    def intra_reduce_scatter(self, tensor, async_op=True): 
        comm_group, group_size = self.get_local_comm_group()
        numel = tensor.numel()
        elem_per_gpu = numel // group_size
        input_splits = [elem_per_gpu for _ in range(group_size)]
        input_splits[-1] = numel - elem_per_gpu * (group_size - 1)
        tensors = list(torch.split(tensor, input_splits))
        ret = torch.empty_like(tensors[self.local_rank])
        if async_op:
            handle = dist.reduce_scatter(ret, tensors, group=comm_group, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.reduce_scatter(ret, tensors, group=comm_group)
        return ret / self.world_size


    def intra_allgather(self, tensor, async_op=True):     
        comm_group, group_size = self.get_local_comm_group()
        ret = [torch.empty_like(tensor) for _ in range(group_size)]
        if async_op:
            handle = dist.all_gather(ret, tensor, group=comm_group, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_gather(ret, tensor, group=comm_group)
        return ret


    # in-place
    def intra_broadcast(self, tensor, src, async_op=True):
        comm_group, _ = self.get_local_comm_group()
        comm_ranks_group = self.get_local_ranks_group()
        # if src is not in the ranks group, it may be the local rank id
        if src not in comm_ranks_group:
            src += self.worker_id * self.local_size
        assert(src in comm_ranks_group)
        if async_op:
            handle = dist.broadcast(tensor, src=src, group=comm_group, async_op=async_op)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.broadcast(tensor, src=src, group=comm_group)
        return tensor


    def global_allgather(self, tensor, async_op=True): 
        ret = [torch.empty_like(tensor) for _ in range(self.world_size)]
        if async_op:
            handle = dist.all_gather(ret, tensor, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_gather(ret, tensor)
        return ret
    

    def global_alltoall(self, tensor, async_op=True): 
        numel = tensor.numel()
        elem_per_gpu = numel // self.world_size
        input_splits = [elem_per_gpu for _ in range(self.world_size)]
        input_splits[-1] = numel - elem_per_gpu * (self.world_size - 1)
        tensors = tensor.split(input_splits)
        ret = [torch.empty_like(tensors[self.rank]) for _ in range(self.world_size)]

        if async_op:
            handle = dist.all_to_all(ret, list(tensors), async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_to_all(ret, list(tensors))
        return ret


    # in-place
    def global_allreduce(self, tensor, async_op=True): 
        if async_op:
            handle = dist.all_reduce(tensor, async_op=True)
            # Wait ensures the operation is enqueued, but not necessarily complete.
            handle.wait()
        else:
            dist.all_reduce(tensor)
        return tensor / self.world_size