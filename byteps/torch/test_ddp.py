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
from mergeComp.communicator.intra_comm_comp import IntraCommComp
from mergeComp.communicator.inter_comm_comp import InterCommComp
from mergeComp.communicator.global_comm_comp import GlobalCommComp

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--local_rank", type=int, 
                        help="Local rank. Necessary for the torch.distributed.launch utility.")
args = parser.parse_args()

has_gpu = True

class TorchTest:
    """
    Tests for communication ops in DDPbackend
    """
    def __init__(self, ddp, dtype=torch.float32):
        # bps is initialized in DDPBackend()
        self.ddp = ddp()
        self.dtypes = self.filter_supported_types([dtype])
        self.intra_comm_comp = IntraCommComp(None, self.ddp)
        self.inter_comm_comp = InterCommComp(None, self.ddp)
        self.global_comm_comp = GlobalCommComp(None, self.ddp)
        self.world_size = bps.size()
        

    def filter_supported_types(self, types):
        return types


    def _current_context(self):
        if has_gpu:
            torch.cuda.set_device(bps.local_rank())
            return torch.device('cuda')
        else:
            return torch.device('cpu')


    def test_intra_gather(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp intra gather:", tensor)
            ret = self.intra_comm_comp.gather([tensor])
            print("tensor after ddp intra gather:", ret)

        print('test_ddp_intra_gather done')
    

    def test_inter_gather(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp inter gather:", tensor, flush=True)
            ret = self.inter_comm_comp.gather([tensor])
            print("tensor after ddp inter gather:", ret, flush=True)

        print('test_ddp_inter_gather done')


    def test_intra_broadcast(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp intra broadcast:", tensor, flush=True)
            ret = self.intra_comm_comp.broadcast([tensor])
            print("tensor after ddp intra broadcast:", ret, flush=True)

        print('test_ddp_intra_broadcast done')


    def test_inter_broadcast(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp inter broadcast:", tensor, flush=True)
            ret = self.inter_comm_comp.broadcast([tensor])
            print("tensor after ddp inter broadcast:", ret, flush=True)

        print('test_ddp_inter_broadcast done')


    def test_allreduce(self):
        dims = [self.world_size]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp allreduce:", tensor, flush=True)
            ret = self.global_comm_comp.allreduce(tensor)
            print("tensor after ddp allreduce:", ret, flush=True)

        print('test_ddp_allreduce done')


    def test_intra_allgather(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp intra allgather:", tensor, flush=True)
            ret = self.intra_comm_comp.allgather([tensor])
            print("tensor after ddp intra allgather:", ret, flush=True)

        print('test_ddp_intra_allgather done')


    def test_inter_allgather(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp inter allgather:", tensor, flush=True)
            ret = self.inter_comm_comp.allgather([tensor])
            print("tensor after ddp inter allgather:", ret, flush=True)

        print('test_ddp_inter_allgather done')

    
    def test_global_allgather(self):
        dims = [2]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp global allgather:", tensor, flush=True)
            ret = self.global_comm_comp.allgather(tensor)
            print("tensor after ddp global allgather:", ret, flush=True)

        print('test_ddp_global_allgather done')

    
    def test_intra_reduce_scatter(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp intra_reduce_scatter:", tensor, flush=True)
            ret = self.intra_comm_comp.reducescatter(tensor)
            print("tensor after ddp intra_reduce_scatter:", ret, flush=True)

        print('test_ddp_intra_reduce_scatter done')
    

    def test_intra_alltoall(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp intra alltoall:", tensor, flush=True)
            ret = self.intra_comm_comp.alltoall([tensor])
            print("tensor after ddp intra alltoall:", ret, flush=True)

        print('test_ddp_intra_alltoall done')


    def test_inter_alltoall(self):
        dims = [8]
        ctx = self._current_context()

        for dtype, dim in itertools.product(self.dtypes, dims):
            tensor = torch.ones(dim, device=ctx, dtype=dtype)
            tensor.mul_(bps.rank() + 1)
            print("tensor before ddp inter alltoall:", tensor, flush=True)
            ret = self.inter_comm_comp.alltoall([tensor])
            print("tensor after ddp inter alltoall:", ret, flush=True)

        print('test_ddp_inter_alltoall done')


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


if __name__ == '__main__':
    DDPbackend = init_comm(args.local_rank)
    dtype = torch.float32
    test = TorchTest(DDPbackend, dtype)
    test.test_intra_gather()
    time.sleep(1)
    test.test_inter_gather()
    time.sleep(1)
    test.test_intra_broadcast()
    time.sleep(1)
    test.test_inter_broadcast()
    time.sleep(1)
    test.test_allreduce()
    time.sleep(1)
    test.test_intra_allgather()
    time.sleep(1)
    test.test_inter_allgather()
    time.sleep(1)
    test.test_global_allgather()
    time.sleep(1)
    test.test_intra_reduce_scatter()
    time.sleep(1)
    test.test_intra_alltoall()
    time.sleep(1)
    test.test_inter_alltoall()
    time.sleep(1)

    # ddp_op_lists = ["allreduce"]
    # #ddp_op_lists = ["allreduce", "allgather", "alltoall", "gather", "broadcast"]
    # for op in ddp_op_lists:
    #     test.test_ddp_goodput(op)
    #     time.sleep(1) 
    # test.test_ddp_broadcast()
    # time.sleep(1) 