# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from byteps.torch.ops import synchronize
import byteps.torch as bps
import itertools
import torch
import os
import numpy as np
import time
from mergeComp.helper import init_comm

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--local_rank", type=int, 
                        help="Local rank. Necessary for the torch.distributed.launch utility.")
args = parser.parse_args()

has_gpu = True

class TorchTest:
    """
    Tests for ops in byteps.torch
    """
    def __init__(self, ddp, dtype=torch.float32):
        # bps is initialized in DDPBackend()
        self.ddp = ddp()
        self.dtypes = self.filter_supported_types([dtype])
        

    def test_ddp_gather(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp gather:", tensor)
            ret = self.ddp.gather(tensor, dst=bps.local_rank())
            print("tensor after ddp gather:", ret)

        print('test_byteps_ddp_gather done')
    

    def test_ddp_local_gather(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp localgather:", tensor)
            ret = self.ddp.local_gather(tensor)
            print("tensor after ddp local gather:", ret)

        print('test_byteps_ddp_gather done')


    def test_ddp_broadcast(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp broadcast:", tensor)
            ret = self.ddp.broadcast(tensor, src=bps.local_rank())
            print("tensor after ddp broadcast:", ret)

        print('test_byteps_ddp_broadcast done')


    def test_ddp_allreduce(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp allreduce:", tensor)
            ret = self.ddp.allreduce(tensor)
            print("tensor after ddp allreduce:", ret)

        print('test_byteps_ddp_allreduce done')


    def test_ddp_allgather(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp allgather:", tensor)
            ret = self.ddp.allgather(tensor)
            print("tensor after ddp allgather:", ret)

        print('test_byteps_ddp_allgather done')

    
    def test_ddp_global_allgather(self):
        dims = [2]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp allgather:", tensor)
            ret = self.ddp.global_allgather(tensor)
            print("tensor after ddp allgather:", ret)

        print('test_byteps_ddp_allgather done')

    
    def test_ddp_intra_reduce_scatter(self):
        dims = [16]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp intra_reduce_scatter:", tensor)
            ret = self.ddp.intra_reduce_scatter(tensor)
            print("tensor after ddp intra_reduce_scatter:", ret)

        print('test_byteps_ddp_intra_reduce_scatter done')


    def test_ddp_intra_allgather(self):
        dims = [4]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp intra_allgather:", tensor)
            ret = self.ddp.intra_allgather(tensor)
            print("tensor after ddp intra_allgather:", ret)

        print('test_byteps_ddp_intra_allgather done')

    
    def test_ddp_intra_broadcast(self):
        dims = [4]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp intra_broadcast:", tensor)
            ret = self.ddp.intra_broadcast(tensor, src=0)
            print("tensor after ddp intra_broadcast:", ret)

        print('test_byteps_ddp_intra_broadcast done')
    

    def test_ddp_alltoall(self):
        dims = [7]
        input_splits = [2, 5]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            tensor = list(tensor.split(input_splits))
            print("tensor before ddp alltoall:", tensor)
            ret = self.ddp.alltoall(tensor)
            print("tensor after ddp alltoall:", ret)

        print('test_byteps_ddp_alltoall done')


    def test_ddp_goodput(self, op):
        sizes = [2**i for i in range(10, 24)]
        ctx = self._current_context()
        count = 100
        comm = None

        if op == "alltoall":
            comm = self.ddp.alltoall
        elif op == "allgather":
            comm = self.ddp.allgather
        elif op == "allreduce":
            comm = self.ddp.allreduce
        elif op == "broadcast":
            comm = self.ddp.broadcast
        elif op == "gather":
            comm = self.ddp.gather

        stat_durations = []
        stat_goodputs = []
        for dtype, size in itertools.product(self.dtypes, sizes):
            tensor = torch.randn((size), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            durations = []
            goodputs = []
            # for _ in range(count):
            #     start_time = time.time()
            #     time_tensor = torch.tensor([start_time], device=ctx, dtype=dtype)
            #     handle = bps.byteps_push_pull(time_tensor, name="inter_comm_sync", average=True)
            #     bps.synchronize(handle)
            #     start_time = time.time()

            #     if op in ["alltoall", "allgather", "allreduce"]:
            #         ret = comm(tensor)
            #     else:
            #         ret = comm(tensor, bps.local_rank())
            #     torch.cuda.synchronize()
            #     end_time = time.time()
            #     time_tensor = torch.tensor([end_time-start_time], device=ctx, dtype=dtype)
            #     end_times = bps.intra_allgather(time_tensor, name="inter_comm_allgather_sync", average=False)
            #     duration = torch.max(end_times).data
            #     durations.append(duration)
            #     goodput = size*4*8*bps.local_size() / (1000*1000*1000) / duration
            #     goodputs.append(goodput)

            # if bps.local_rank() == 0:
            #     print(op, "local rank:", bps.local_rank(), "local size:", bps.local_size(), "tensor size:", size*4, "duration:", sum(durations)/len(durations), " goodput:", sum(goodputs)/len(goodputs), "Gbps", flush=True)
            #     stat_goodputs.append(sum(goodputs)/len(goodputs))
            #     stat_durations.append(sum(durations)/len(durations))
            
            start_time = time.time()
            time_tensor = torch.tensor([start_time], device=ctx, dtype=dtype)
            handle = bps.byteps_push_pull(time_tensor, name="inter_comm_sync", average=True)
            bps.synchronize(handle)
            start_time = time.time()
            for _ in range(count):
                if op in ["alltoall", "allgather", "allreduce"]:
                    ret = comm(tensor)
                else:
                    ret = comm(tensor, bps.local_rank())
            torch.cuda.synchronize()
            end_time = time.time()
            # time_tensor = torch.tensor([end_time-start_time], device=ctx, dtype=dtype)
            # end_times = bps.intra_allgather(time_tensor, name="inter_comm_allgather_sync", average=False)
            # duration = (torch.max(end_times).cpu().item())/count
            duration = (end_time-start_time)/count
            durations.append(duration)
            data_size = 16 if dtype in [torch.float16, torch.int16] else 32  
            goodput = size*bps.local_size() / (1000*1000*1000) / duration

            if bps.local_rank() == 0:
                print(op, "local rank:", bps.local_rank(), "local size:", bps.local_size(), "tensor size:", size*4, "duration:", duration, " goodput:", goodput, "Gbps", flush=True)
                stat_goodputs.append(round(goodput, 4))
                stat_durations.append(round(duration, 6))

        if bps.local_rank() == 0:
            print(op, stat_goodputs)
            print(op, stat_durations)


        #print('test_byteps_intra_allgather_goodput done')

    def _current_context(self):
        if has_gpu:
            torch.cuda.set_device(bps.local_rank())
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def filter_supported_types(self, types):
        return types

    def test_byteps_intra_gather(self):
        """Test that the intra_gather correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(1), (17)]
        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones((4), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            print("tensor before intra gather:", tensor)
            handle = bps.intra_gather_async(tensor, name="tensor_gather"+str(count), average=False, root=0)
            tensor = bps.synchronize(handle)

            print("tensor after intra gather:", tensor)

        print('test_byteps_intra_gather done')


    def test_byteps_intra_broadcast(self):
        """Test that the intra_gather correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(1), (17)]
        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones((4), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            print("tensor before intra broadcast:", tensor)
            handle = bps.intra_broadcast_async(tensor, name="tensor_broadcast"+str(count), average=False, root=0)
            tensor = bps.synchronize(handle)

            print("tensor after intra broadcast:", tensor)

        print('test_byteps_intra_broadcast done')


    def test_byteps_intra_reducescatter(self):
        """Test that the intra_gather correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(1), (17)]
        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones((10), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            print("tensor before intra reducescatter:", tensor)
            handle = bps.intra_reducescatter_async(tensor, name="tensor_reducescatter"+str(count), average=False)
            tensor = bps.synchronize(handle)
            print("tensor after intra reducescatter:", tensor)

        print('test_byteps_intra_reducescatter done')


    def test_byteps_intra_allgather(self):
        """Test that the intra_allgather correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(1), (17)]
        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones((1), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            #print("tensor before intra allgather:", tensor)
            handle = bps.intra_allgather_async(tensor, name="tensor_allgather"+str(count), average=False)
            tensor = bps.synchronize(handle)
            #print("tensor after intra allgather:", tensor)

        print('test_byteps_intra_allgather done')

    
    def test_byteps_goodput(self, op="reducescatter"):
        sizes = [2**i for i in range(8, 24)]
        ctx = self._current_context()
        count = 100

        comm = bps.intra_reducescatter_async
        if op == "intra_allgather":
            comm = bps.intra_allgather_async
        elif op == "intra_gather_async":
            comm = bps.intra_gather_async
        elif op == "intra_broadcast":
            comm = bps.intra_broadcast_async
        elif op == "intra_alltoall":
            comm = bps.intra_alltoall_async
        elif op == "push_pull":
            comm = bps.byteps_push_pull

        goodputs = []
        latencies = []
        for dtype, size in itertools.product(self.dtypes, sizes):
            tensor = torch.ones((size), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            torch.cuda.synchronize()
            start_time = time.time()
            handles = []
            for i in range(count):
                handle = comm(tensor, name=op+str(size)+str(i), average=False)
                handles.append(handle)
            for handle in handles:
                synchronize(handle)
            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time
            data_size = 16 if dtype in [torch.float16, torch.int16] else 32  
            goodput = size*count*data_size/duration/10e8
            goodputs.append(round(goodput, 4))
            latencies.append(round(duration/count, 6))
            if bps.rank() == 0:
                print(op, "local size:", bps.local_size(), "tensor size:", size*4, "Bytes, comm time:", duration/count, " goodput:", goodput, "Gbps")
        if bps.rank() == 0:
            print(op, goodputs)
            print(op, latencies)


        #print('test_byteps_intra_allgather_goodput done')


    def test_byteps_intra_alltoall(self):
        """Test that the intra_alltoall correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(1), (17)]
        for i in range(50):
            for dtype, dim in itertools.product(self.dtypes, dims):
                tensor = torch.ones((1610720), device=ctx, dtype=dtype)
                tensor.mul_(bps.local_rank() + 1)
                #print("tensor before intra alltoall:", tensor)
                handle = bps.intra_alltoall_async(tensor, name="tensor_alltoall"+str(i), average=False)
                handle2 = bps.intra_alltoall_async(tensor, name="tensor_alltoall2"+str(i), average=False)
                tensor = bps.synchronize(handle)
                tensor2 = bps.synchronize(handle2)
                #print("tensor after intra alltoall:", tensor)

        print('test_byteps_intra_alltoall done')

    def test_byteps_intra_reduce(self):
        """Test that the intra_reduce correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(1), (17)]
        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones((4), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            print("tensor before intra reduce:", tensor)
            bps.intra_reduce(tensor, name="tensor_reduce"+str(count), average=False)
            print("tensor after intra reduce:", tensor)

        print('test_byteps_intra_reduce done')


    def test_byteps_push_pull(self):
        """Test that the intra_reduce correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(1), (17)]
        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones((8), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            print("tensor before push pull:", tensor)
            handle = bps.byteps_push_pull(tensor, name="tensor_push_pull"+str(count), average=True)
            tensor = synchronize(handle)
            print("tensor after push pull:", tensor)

        print('test_byteps_push_pull done')


    def test_byteps_inter_compress(self):
        """Test that the intra_reduce correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(1), (17)]
        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones((2), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            print("tensor before inter compress:", tensor)
            if bps.local_rank() == 2:
                bps.inter_compress(tensor, name="tensor_inter_compress"+str(count), average=False, role=bps.local_rank())
            else:
                bps.inter_compress(tensor, name="tensor_inter_compress"+str(count), average=False, role=-1)
            print("tensor after inter compress:", tensor)

        print('test_inter_compress done')

    def test_byteps_cpu_compress(self):
        """Test that the intra_reduce correctly sums 1D, 2D, 3D tensors."""
        dims = [1]
        ctx = self._current_context()
        shapes = [(1), (17)]
        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones((64), device=ctx, dtype=dtype)
            tensor.mul_(bps.local_rank() + 1)
            print("tensor before cpu compress:", tensor)
            bps.cpu_compress(tensor, name="tensor_cpu_compress", average=False)
            print("tensor after cpu compress:", tensor)

        print('test_cpu_compress done')


if __name__ == '__main__':
    DDPbackend = init_comm(args.local_rank)
    dtype = torch.float32
    test = TorchTest(DDPbackend, dtype)
    # test.test_byteps_intra_gather()
    # time.sleep(2)
    # test.test_byteps_intra_reduce()
    # time.sleep(2)
    # test.test_byteps_inter_compress()
    # time.sleep(1)
    # test.test_byteps_cpu_compress()
    # time.sleep(1)
    # test.test_byteps_push_pull()
    # time.sleep(1)
    # test.test_byteps_intra_allgather()
    # time.sleep(1)
    # test.test_byteps_intra_alltoall()
    # time.sleep(1)
    # test.test_byteps_intra_broadcast()
    # time.sleep(1)
    # test.test_byteps_intra_reducescatter()
    # time.sleep(1) 
    # test.test_ddp_gather()
    # time.sleep(1) 
    # test.test_ddp_local_gather()
    # time.sleep(1) 
    # test.test_ddp_local_broadcast()
    # time.sleep(1)
    # test.test_ddp_allreduce()
    # time.sleep(1) 
    test.test_ddp_global_allgather()
    time.sleep(1) 
    test.test_ddp_intra_reduce_scatter()
    time.sleep(1) 
    test.test_ddp_intra_allgather()
    time.sleep(1) 
    test.test_ddp_intra_broadcast()
    time.sleep(1) 
    # test.test_ddp_allgather()
    # time.sleep(1) 
    # test.test_ddp_alltoall()
    # time.sleep(1) 

    # byteps_op_lists = ["intra_alltoall"]
    # byteps_op_lists = ["reducescatter", "intra_gather", "intra_broadcast", "intra_alltoall", "push_pull", "intra_allgather"]
    # for op in byteps_op_lists:
    #     test.test_byteps_goodput(op)
    #     time.sleep(1) 
    
    # ddp_op_lists = ["allreduce"]
    # #ddp_op_lists = ["allreduce", "allgather", "alltoall", "gather", "broadcast"]
    # for op in ddp_op_lists:
    #     test.test_ddp_goodput(op)
    #     time.sleep(1) 
    # test.test_ddp_broadcast()
    # time.sleep(1) 