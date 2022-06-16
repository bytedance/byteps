# Copyright 2021 Bytedance Inc. All Rights Reserved.
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

import os
import torch.distributed as dist
from typing import Optional, List, Any, Tuple, overload
from torch import Tensor
import collections

from byteps.common import BytePSBasics as _BytePSBasics
import byteps
bps_path=os.path.dirname(byteps.__file__) + '/torch/'
_basics = _BytePSBasics(bps_path, 'c_lib')


bps_init = _basics.init
bps_shutdown = _basics.shutdown

from torch._C._distributed_c10d import (
    AllreduceOptions,
    AllreduceCoalescedOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    PrefixStore,
    ProcessGroup,
    ReduceOptions,
    ReduceOp,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
)
from torch._C._distributed_c10d import AllgatherOptions as AllGatherOptions

from datetime import timedelta
import atexit

class Work:
    def is_completed(self) -> bool:
        pass
    def is_success(self) -> bool:
        pass
    def exception(self) -> Any:
        paiss
    def wait(self, timeout: timedelta = timedelta()) -> bool:
        pass
    def source_rank(self) -> int:
        pass
    def _source_rank(self) -> int:
        pass
    def result(self) -> List[Tensor]:
        pass
    def synchronize(self):
        pass

def is_joint_mode():
    return os.getenv("BYTEPS_JOINT_MODE", "1").lower() in ["1", "true"]

def is_a100_node():
    return "A100" in os.getenv("ML_PLATFORM_DEVICE_TYPE", "")

def _dump_byteps_relevant_env_vars(desc=None, envars=None):
    relevant_env_vars = [
        "MASTER_PORT",
        "MASTER_ADDR",
        "CUDA_VISIBLE_DEVICES",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "DMLC_WORKER_ID",
        "BYTEPS_NODE_ID",
        "BYTEPS_PARTITION_BYTES",
        "BYTEPS_REDUCE_ROOTS",
        # ps-lite envars
        "DMLC_ROLE",
        "DMLC_ENABLE_RDMA",
        "BYTEPS_ENABLE_IPC",
        "DMLC_ENABLE_UCX",
        "UCX_SOCKADDR_CM_ENABLE",
        "UCX_RDMA_CM_SOURCE_ADDRESS",
        "DMLC_NODE_HOST",
        "DMLC_NUM_WORKER",
        "DMLC_NUM_SERVER",
        "DMLC_PS_ROOT_URI",
        "DMLC_PS_ROOT_PORT",
        "PS_VERBOSE",
        "GLOG_log_dir",
        "GLOG_logtostderr",
    ]
    formatted_output = ""
    if desc:
        formatted_output += "%s\n" % (desc)

    if envars is None:
        envars = os.environ
    for var in relevant_env_vars:
        value = envars[var] if var in envars else "N/A"
        formatted_output += "env:%s=%s\n" % (var, value)
    print(formatted_output)

class ProcessGroupBYTEPS():
    byteps_inited = False
    server_proc = None
    scheduler_proc = None

    def __init__(
        self,
        store: dist.Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ):
        self._rank = rank
        self._size = size
        self._counter = collections.defaultdict(int)
        self._dup_limit = int(os.getenv('BYTEPS_DUP_NAME_LIMIT', 100))

        self._skip_allgather = os.getenv('BYTEPS_DEBUG_SKIP_DIST_ALLGATHER', '').lower() in ["1"]
        self._simulate_allgather = int(os.getenv('BYTEPS_DEBUG_SIMULATE_ALLGATHER', '0'))

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def _simulated_allreduce(
        self,
        tensors: List[Tensor],
        opts: AllreduceOptions = AllreduceOptions(),
    ) -> Work:
        assert isinstance(tensors, list), type(tensors)
        assert len(tensors) == 1, "tensors must have exactly one entry"

        op = opts.reduceOp

        import torch
        from byteps.torch.ops import allgather as byteps_allgather
        input_tensor = tensors[0]
        input_shape_str = '_'.join([str(x) for x in list(input_tensor.size())])
        base_name = f'adhoc_allreduce_{input_tensor.dtype}_{input_tensor.numel()}_{input_shape_str}'
        counter = self._counter[base_name] % self._dup_limit
        name = f'{base_name}_{counter}'
        self._counter[base_name] += 1

        output = byteps_allgather(input=input_tensor, name=name)
        output_tensors = []
        start = 0
        item = input_tensor
        for index in range(self.size()):
            length = item.shape[0]
            output_tensors.append(output.narrow(0, start, length))
            start += length

        if op == ReduceOp.MIN:
            result = torch.min(torch.stack(output_tensors), dim=0)[0]
        elif op == ReduceOp.MAX:
            result = torch.max(torch.stack(output_tensors), dim=0)[0]
        else:
            raise ValueError(f'Unsupported operation, allreduce(op={op})')
        input_tensor.copy_(result)
        return Work()

    def allreduce(
        self,
        tensors: List[Tensor],
        opts: AllreduceOptions = AllreduceOptions(),
    ) -> Work:
        from byteps.torch.ops import push_pull_async_inplace
        from byteps.torch.ops import synchronize

        if opts.reduceOp in [ReduceOp.MAX, ReduceOp.MIN]:
            return self._simulated_allreduce(tensors, opts)

        assert opts.reduceOp == ReduceOp.SUM, "Only ReduceOp.SUM is supported"
        average = opts.reduceOp != ReduceOp.SUM
        handles = []
        for item in tensors:
            shape_str = '_'.join([str(x) for x in list(item.size())])
            base_name = f'adhoc_allreduce_{item.dtype}_{item.numel()}_{shape_str}'
            counter = self._counter[base_name] % self._dup_limit
            name = f'{base_name}_{counter}'
            handle = push_pull_async_inplace(item, average=average, name=name)
            self._counter[base_name] += 1
            handles.append(handle)
        for handle in handles:
            synchronize(handle)
        return Work()

    def broadcast(
        self,
        tensor: Tensor,
        root: int,
    ) -> Work:
        raise NotImplementedError

    def broadcast(
        self,
        tensors: List[Tensor],
        opts = BroadcastOptions(),
    ) -> Work:
        from byteps.torch.functions import broadcast_parameters as bps_bcast

        root_rank = opts.rootRank
        tensors_with_name = []
        for item in tensors:
            shape_str = '_'.join([str(x) for x in list(item.size())])
            base_name = f'adhoc_broadcast_{item.dtype}_{item.numel()}_{shape_str}'
            counter = self._counter[base_name] % self._dup_limit
            self._counter[base_name] += 1
            name = f'{base_name}_{counter}'
            tensors_with_name.append((name, item))
        bps_bcast(tensors_with_name, root_rank, prefix='byteps.dist.')
        return Work()

    def allreduce_coalesced(
        self,
        tensors: List[Tensor],
        opts = AllreduceCoalescedOptions(),
    ) -> Work:
        raise NotImplementedError

    def reduce(
        self,
        tensor: Tensor,
        root: int,
        op = ReduceOp.SUM,
    ) -> Work:
        raise NotImplementedError

    def reduce(
        self,
        tensors: List[Tensor],
        opts = ReduceOptions(),
    ) -> Work:
        assert isinstance(tensors, list), type(tensors)
        assert len(tensors) == 1, "tensors must have exactly one entry"

        op = opts.reduceOp
        dst = opts.rootRank

        import torch
        from byteps.torch.ops import allgather as byteps_allgather
        input_tensor = tensors[0]
        input_shape_str = '_'.join([str(x) for x in list(input_tensor.size())])
        base_name = f'adhoc_reduce_{input_tensor.dtype}_{input_tensor.numel()}_{input_shape_str}'
        counter = self._counter[base_name] % self._dup_limit
        name = f'{base_name}_{counter}'
        self._counter[base_name] += 1

        output = byteps_allgather(input=input_tensor, name=name)
        output_tensors = []
        start = 0
        item = input_tensor
        for index in range(self.size()):
            length = item.shape[0]
            output_tensors.append(output.narrow(0, start, length))
            start += length

        if op == ReduceOp.MIN:
            result = torch.min(torch.stack(output_tensors), dim=0)[0]
        elif op == ReduceOp.MAX:
            result = torch.max(torch.stack(output_tensors), dim=0)[0]
        elif op == ReduceOp.SUM:
            result = torch.sum(torch.stack(output_tensors), dim=0)
        else:
            raise ValueError(f'Unsupported operation, reduce(op={op})')
        input_tensor.copy_(result)
        return Work()

    def _simulated_allgather(
            self,
            input_tensor: Tensor,
            name: str,
    ) -> Tensor:
        import torch
        from byteps.torch.ops import push_pull_async_inplace, synchronize

        if input_tensor.dim() == 0:
            input_tensor = torch.unsqueeze(input_tensor, 0)
        shape = list(input_tensor.shape)
        start = self._rank * shape[0]
        length = shape[0]
        shape[0] *= self._size
        output_tensor = input_tensor.new_zeros(torch.Size(shape))
        output_tensor.narrow(0, start, length).copy_(input_tensor)
        handle = push_pull_async_inplace(output_tensor, average=False, name=name)
        synchronize(handle)
        return output_tensor

    def allgather(
        self,
        output_tensors: List[Tensor],
        input_tensor: Tensor,
    ) -> Work:
        raise NotImplementedError

    def allgather(
        self,
        output_tensors: List[List[Tensor]],
        input_tensors: List[Tensor],
        opts = AllGatherOptions(),
    ) -> Work:
        if self._skip_allgather:
            return Work()

        # check input_tensors is a list and contains only one tensor
        assert isinstance(input_tensors, list), type(input_tensors)
        assert isinstance(output_tensors, list), type(output_tensors)
        assert len(input_tensors) == 1, "input_tensors must have exactly one entry"

        import torch
        from byteps.torch.ops import allgather as byteps_allgather
        output_tensors = output_tensors[0]
        input_tensor = input_tensors[0]

        input_shape_str = '_'.join([str(x) for x in list(input_tensor.size())])
        base_name = f'adhoc_allgather_{input_tensor.dtype}_{input_tensor.numel()}_{input_shape_str}'
        counter = self._counter[base_name] % self._dup_limit
        name = f'{base_name}_{counter}'
        self._counter[base_name] += 1

        if input_tensor.numel() * input_tensor.element_size() <= self._simulate_allgather:
            output = self._simulated_allgather(input_tensor=input_tensor, name=name)
        else:
            output = byteps_allgather(input=input_tensor, name=name)
        start = 0
        for index, item in enumerate(output_tensors):
            if len(item.shape) == 0:
                item = torch.unsqueeze(item, 0)
            length = item.shape[0]
            item.copy_(output.narrow(0, start, length))
            start += length
        return Work()

    def _allgather_base(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor
    ) -> Work:
        # check input_tensors is a list and contains only one tensor
        assert isinstance(output_tensor, Tensor)
        assert isinstance(input_tensor, Tensor)

        from byteps.torch.ops import allgather as byteps_allgather
        input_shape_str = '_'.join([str(x) for x in list(input_tensor.size())])
        base_name = f'adhoc_allgather_base_{input_tensor.dtype}_{input_tensor.numel()}_{input_shape_str}'
        counter = self._counter[base_name] % self._dup_limit
        name = f'{base_name}_{counter}'
        self._counter[base_name] += 1

        byteps_allgather(input=input_tensor, output=output_tensor, name=name)
        return Work()

    def allgather_coalesced(
        self,
        output_lists: List[List[Tensor]],
        input_list: List[Tensor],
        opts = AllGatherOptions(),
    ) -> Work:
        raise NotImplementedError

    def gather(
        self,
        output_tensors: List[List[Tensor]],
        input_tensors: List[Tensor],
        opts = GatherOptions(),
    ) -> Work:
        raise NotImplementedError

    def gather(
        self,
        output_tensors: List[Tensor],
        input_tensor: Tensor,
        root: int,
    ) -> Work:
        raise NotImplementedError

    def scatter(
        self,
        output_tensors: List[Tensor],
        input_tensors: List[List[Tensor]],
        opts = ScatterOptions(),
    ) -> Work:
        raise NotImplementedError

    def scatter(
        self,
        output_tensor: Tensor,
        input_tensors: List[Tensor],
        root: int,
    ) -> Work:
        raise NotImplementedError

    def reduce_scatter(
        self,
        output_tensors: Tensor,
        input_tensor: List[Tensor],
    ) -> Work:
        raise NotImplementedError

    def reduce_scatter(
        self,
        output_tensors: List[Tensor],
        input_tensors: List[List[Tensor]],
        opts = ReduceScatterOptions(),
    ) -> Work:
        assert isinstance(output_tensors, list), type(output_tensors)
        assert len(output_tensors) == 1, "output_tensors must have exactly one entry"
        assert isinstance(input_tensors, list), type(input_tensors)
        assert len(input_tensors) == 1, "input_tensors must have exactly one entry"

        from byteps.torch.ops import _do_push_pull_async
        from byteps.torch.ops import synchronize

        assert opts.reduceOp == ReduceOp.SUM, "Only ReduceOp.SUM is supported"
        average = opts.reduceOp != ReduceOp.SUM
        handles = []
        input_tensors = input_tensors[0]
        for idx, item in enumerate(input_tensors):
            shape_str = '_'.join([str(x) for x in list(item.size())])
            base_name = f'adhoc_reduce_scatter_{item.dtype}_{item.numel()}_{shape_str}'
            counter = self._counter[base_name] % self._dup_limit
            name = f'{base_name}_{counter}'
            if idx == self._rank:
                output = output_tensors[0]
            else:
                output = item
            handle = _do_push_pull_async(item, output, average=average, name=name)
            self._counter[base_name] += 1
            handles.append(handle)
        for handle in handles:
            synchronize(handle)
        return Work()

    def alltoall_base(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts = AllToAllOptions(),
    ) -> Work:
        raise NotImplementedError

    def alltoall_base(
        self,
        output: Tensor,
        input: Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> Work:
        raise NotImplementedError

    def alltoall(
        self,
        output_tensor: List[Tensor],
        input_tensor: List[Tensor],
        opts = AllToAllOptions(),
    ) -> Work:
        raise NotImplementedError

    def alltoall(
        self,
        output: List[Tensor],
        input: List[Tensor],
    ) -> Work:
        raise NotImplementedError

    def send(
        self,
        tensors: List[Tensor],
        dstRank: int,
        tag: int,
    ) -> Work:
        raise NotImplementedError

    def recv(
        self,
        tensors: List[Tensor],
        srcRank: int,
        tag: int,
    ) -> Work:
        raise NotImplementedError

    def recv_anysource(
        self,
        tensors: List[Tensor],
        tag: int
    ) -> Work:
        raise NotImplementedError

    def barrier(
        self,
        opts = BarrierOptions()
    ) -> Work:
        import torch
        from byteps.torch.ops import push_pull

        root_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.zeros(1, device=device)
        push_pull(tensor, name='ad_hoc.barrier.')
        return Work()
    def _get_backend_name(self):
        return 'nccl'

def execute_cmd(cmd, env=None):
    import subprocess
    import sys

    make_process = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, shell=True)
    return make_process

def shutdown_scheduler():
    proc = ProcessGroupBYTEPS.scheduler_proc
    if not proc:
        return
    proc.communicate()
    if proc.returncode:
        print(f'An ERROR occured while running the BytePS scheduler\n'
              f'Exit code: {proc.returncode}')

def shutdown_server():
    proc = ProcessGroupBYTEPS.server_proc
    if not proc:
        return
    proc.communicate()
    if proc.returncode:
        print(f'An ERROR occured while running the BytePS server\n'
              f'Exit code: {proc.returncode}')

def start_scheduler_server():
    dmlc_num_worker = int(os.environ["DMLC_NUM_WORKER"])
    byteps_num_node = int(os.environ['WORLD_SIZE']) // int(os.environ['BYTEPS_LOCAL_SIZE'])
    force_distributed = int(os.getenv("BYTEPS_FORCE_DISTRIBUTED", "0"))
    if byteps_num_node == 1 and not force_distributed:
        return

    if os.environ['ML_PLATFORM_NODE_ID'] == '0' and int(os.environ['LOCAL_RANK']) == 0:
        # start the scheduler
        my_env = os.environ.copy()
        my_env["DMLC_ROLE"] = "scheduler"
        my_env["DMLC_NODE_HOST"] = my_env["DMLC_PS_ROOT_URI"]

        if os.getenv('BYTEPS_DEBUG_DUMP_ENVARS'):
            _dump_byteps_relevant_env_vars(desc="byteps_scheduler", envars=my_env)
        cmd = "python3 -c 'import byteps.server'"
        ProcessGroupBYTEPS.scheduler_proc = execute_cmd(cmd, env=my_env)
        atexit.register(shutdown_scheduler)

    if os.environ['DMLC_ROLE'] != 'joint' and os.getenv('ML_PLATFORM_SERVER_HOSTS') is None and int(os.environ['LOCAL_RANK']) == 0:
        # start the server
        my_env = os.environ.copy()
        my_env["DMLC_ROLE"] = "server"

        if os.getenv('BYTEPS_DEBUG_DUMP_ENVARS'):
            _dump_byteps_relevant_env_vars(desc="byteps_server", envars=my_env)
        cmd = "python3 -c 'import byteps.server'"
        ProcessGroupBYTEPS.server_proc = execute_cmd(cmd, env=my_env)
        atexit.register(shutdown_server)

def _parse_nv_topo(nv_topo_str):
    res = []
    for i, line in enumerate(nv_topo_str.splitlines()):
        if not line.startswith('GPU'):
            continue
        line = line.strip().split()
        if i > 0 and i < 9:
            res.append(line[-3])

    def get_best_gpu(x):
        if x in [0, 1]: return 2
        if x in [2, 3]: return 0
        if x in [4, 5]: return 6
        if x in [6, 7]: return 4

    for i in range(8):
        if res[i] == 'PIX':
            return get_best_gpu(i)

    return 0

def _get_v100_reduce_roots():
    if os.environ['BYTEPS_LOCAL_SIZE'] != "8":
        return "0"

    import subprocess
    cmd = 'nvidia-smi topo -m'
    nv_topo_str = subprocess.check_output(cmd, shell=True, encoding='UTF-8').strip()
    reduce_root = _parse_nv_topo(nv_topo_str)
    return str(reduce_root)

def _get_ml_platform_worker_0_ip():
    import socket
    return socket.gethostbyname(os.environ['ML_PLATFORM_WORKER_0_HOST'])

def _get_num_nic():
    import subprocess
    cmd = "nvidia-smi topo -m | grep mlx | grep PXB | wc -l"
    return int(subprocess.check_output(cmd, shell=True, encoding='UTF-8').strip())

def _get_ip_by_nic_idx(nic_idx):
    import subprocess
    cmd = f"ip -4 addr show eth{nic_idx} | awk '/inet/ {{print $2}}' | cut -d/ -f1"
    return subprocess.check_output(cmd, shell=True, encoding='UTF-8').strip()

def _get_nic_ip(local_rank):
    import subprocess
    cmd = "nvidia-smi topo -m | grep mlx | grep PXB | wc -l"
    num_nic = _get_num_nic()
    nic_idx = local_rank // (8 // num_nic)
    return _get_ip_by_nic_idx(nic_idx)


def _get_default_ip():
    import subprocess
    cmd = f"hostname -I | cut -d' ' -f1"
    return subprocess.check_output(cmd, shell=True, encoding='UTF-8').strip()

def _get_numa_id(local_rank):
    import subprocess

    cmd = 'nvidia-smi topo -m'
    nv_topo_str = subprocess.check_output(cmd, shell=True, encoding='UTF-8').strip()
    res = []
    for i, line in enumerate(nv_topo_str.splitlines()):
        if not line.startswith('GPU'):
            continue
        line = line.strip().split()
        if i > 0 and i < 9:
            res.append(line[-1])
    return str(res[local_rank])


def _set_default_env(name, val):
    assert isinstance(name, str)
    if not isinstance(val, str):
        val = str(val)

    if not os.getenv(name, ''):
        os.environ[name] = val

def _set_core_affinity_a100():
    # set core affinity
    should_set_core_affinity = int(os.getenv("TORCH_BYTECCL_CORE_AFFINITY", "1"))
    if not should_set_core_affinity:
        return

    core_list = {
            0: list(range(0, 16)),
            1: list(range(16, 32)),
            2: list(range(64, 80)),
            3: list(range(80, 96)),
            4: list(range(32, 48)),
            5: list(range(48, 64)),
            6: list(range(96, 112)),
            7: list(range(112, 128)),
    }
    local_rank = int(os.environ['BYTEPS_LOCAL_RANK'])
    core_list = core_list[local_rank]
    os.sched_setaffinity(0, core_list)


def _set_core_affinity_v100():
    # set core affinity
    should_set_core_affinity = int(os.getenv("TORCH_BYTECCL_CORE_AFFINITY", "1"))
    if not should_set_core_affinity:
        return

    local_rank = int(os.environ['BYTEPS_LOCAL_RANK'])
    import multiprocessing as mp
    core_count = mp.cpu_count()
    cores_per_proc = int(os.getenv("CORES_PER_PROC", "12"))
    core_blacklist = os.getenv("CORE_BLACKLIST", "-1")
    core_blacklist = [int(item) for item in core_blacklist.split(",")]
    print(f"blacklist {core_blacklist}")

    phy_core_count = core_count // 2
    phy_cores_per_proc = cores_per_proc // 2

    full_list = list(range(0, phy_core_count))
    core_list = [
            item for item in full_list if item not in core_blacklist
    ]
    core_list = [
            core_list[x : x + phy_cores_per_proc]
            for x in range(0, len(core_list), phy_cores_per_proc)
    ]
    for idx, li in enumerate(core_list):
        core_list[idx] = li + [x + phy_core_count for x in li]
    if local_rank == 0:
        print("core_list:")
        for idx, item in enumerate(core_list):
            print(f"    {idx}: {item}")
    core_list = core_list[local_rank]
    should_segregate_cores = int(os.getenv("SEGREGATE_CORES", "0"))
    bps_core_list = (
            core_list[0:2]
            + core_list[phy_cores_per_proc : phy_cores_per_proc + 2]
    )
    torch_core_list = [
            xx for xx in core_list if xx not in bps_core_list
    ]
    if should_set_core_affinity:
        print(
                f"lrank: {local_rank} setting " f"core affinity"
                )
        print(
                f"lrank: {local_rank} my_core_list:" f" {core_list}"
        )
        os.sched_setaffinity(0, core_list)

def _new_byteps_process_group_hook(
        prefix_store, rank, world_size, timeout
        ):
    # setup some byteps specific ENVs here
    os.environ['BYTEPS_LOCAL_RANK'] = os.environ['LOCAL_RANK']
    if 'LOCAL_WORLD_SIZE' in os.environ:
        os.environ['BYTEPS_LOCAL_SIZE'] = os.environ['LOCAL_WORLD_SIZE']
    elif 'BYTEPS_LOCAL_SIZE' not in os.environ:
        raise ValueError(f'Missing environment variable LOCAL_WORLD_SIZE')

    if 'GROUP_RANK' in os.environ:
        os.environ['DMLC_WORKER_ID'] = os.environ['GROUP_RANK']
    elif 'DMLC_WORKER_ID' not in os.environ:
        raise ValueError(f'Missing environment variable GROUP_RANK')

    os.environ['BYTEPS_DISABLE_P2P'] = os.getenv('BYTEPS_DISABLE_P2P', '1')
    os.environ['BYTEPS_DISABLE_CPU_ALLREDUCE'] = os.getenv('BYTEPS_DISABLE_CPU_ALLREDUCE', '0')

    # ps-lite specific envars
    os.environ['DMLC_ENABLE_RDMA'] = os.getenv('DMLC_ENABLE_RDMA', '1')
    os.environ['BYTEPS_ENCODING_SCHEME_VERSION'] = os.getenv('BYTEPS_ENCODING_SCHEME_VERSION', '1')

    dmlc_num_worker = int(os.environ['WORLD_SIZE']) // int(os.environ['BYTEPS_LOCAL_SIZE'])
    num_phy_node = dmlc_num_worker
    if is_a100_node():
        _set_default_env('BYTEPS_PARTITION_BYTES', 32_768_000)
    elif num_phy_node == 1:
        os.environ['BYTEPS_PARTITION_BYTES'] = str(512_208_896)
    os.environ['DMLC_NUM_WORKER'] = str(dmlc_num_worker)
    dmlc_num_server = str(dmlc_num_worker)
    if os.getenv('ML_PLATFORM_SERVER_HOSTS') is not None:
        dmlc_num_server = os.getenv('ML_PLATFORM_SERVER_NUM')
    os.environ['DMLC_NUM_SERVER'] = dmlc_num_server
    os.environ['DMLC_ROLE'] = 'worker'
    master_addr = os.getenv('MASTER_ADDR', '')
    dmlc_ps_root_uri = os.getenv('DMLC_PS_ROOT_URI', "")
    os.environ['DMLC_PS_ROOT_URI'] = master_addr
    if len(dmlc_ps_root_uri):
        os.environ['DMLC_PS_ROOT_URI'] = dmlc_ps_root_uri
    assert len(os.environ['DMLC_PS_ROOT_URI']) > 0, "DMLC_PS_ROOT_URI is required"
    master_port = os.getenv('MASTER_PORT', '')
    dmlc_ps_root_port = os.getenv('DMLC_PS_ROOT_PORT', '')
    os.environ['DMLC_PS_ROOT_PORT'] = master_port
    if len(dmlc_ps_root_port):
        os.environ['DMLC_PS_ROOT_PORT'] = dmlc_ps_root_port
    os.environ['BYTEPS_ADDRESS_POOL_SIZE'] = os.getenv('BYTEPS_ADDRESS_POOL_SIZE', '20480')
    dmlc_node_host = os.getenv('DMLC_NODE_HOST', '').strip()
    if len(dmlc_node_host) == 0:
        eth_ip = _get_default_ip()
        _set_default_env('DMLC_NODE_HOST', eth_ip)

    if os.environ['DMLC_ROLE'] == 'worker' and 'BYTEPS_REDUCE_ROOTS' not in os.environ:
        if is_a100_node() and _get_num_nic() == 2:
            if os.environ['BYTEPS_LOCAL_SIZE'] == "8":
                os.environ['BYTEPS_REDUCE_ROOTS'] = "3,7"
            else:
                print("WARNING: On A100 nodes with two NICs, BytePS is only highly optimized for 8 GPUs")
        elif not is_a100_node():
            os.environ['BYTEPS_REDUCE_ROOTS'] = _get_v100_reduce_roots()
    print("BYTEPS_REDUCE_ROOTS is: ", os.getenv('BYTEPS_REDUCE_ROOTS', ''))

    if is_joint_mode() or is_a100_node():
        os.environ['DMLC_WORKER_ID'] = os.environ['RANK']
        os.environ["DMLC_NUM_WORKER"] = os.environ['WORLD_SIZE']
        os.environ["DMLC_NUM_SERVER"] = os.environ['WORLD_SIZE']
        os.environ["BYTEPS_ENABLE_IPC"] = os.getenv('BYTEPS_ENABLE_IPC', '1')
        os.environ['DMLC_ROLE'] = 'joint'
        if is_a100_node():
            os.environ['DMLC_PS_ROOT_URI'] = _get_ml_platform_worker_0_ip()
            local_rank = int(os.environ['BYTEPS_LOCAL_RANK'])
            os.environ['DMLC_NODE_HOST'] = _get_nic_ip(local_rank)
            os.environ['BYTEPS_NUMA_ID'] = _get_numa_id(local_rank)
            os.environ['BYTEPS_PHY_NODE_ID'] = os.environ['ML_PLATFORM_NODE_ID']

            _set_default_env('BYTEPS_USE_GDR_ALLREDUCE', 1)
            os.environ['DMLC_ENABLE_RDMA'] = 'ibverbs'
            _set_default_env('BYTEPS_BE_ENABLE_POLLING', 1)
            _set_default_env('BYTEPS_BE_WORKER_NUM', 1)
            _set_default_env('BYTEPS_BE_POLL_RETRY_COUNT', 32)
            os.environ["BYTEPS_ENABLE_IPC"] = '0'
    else:
        os.environ['BYTEPS_ENABLE_IPC'] = os.getenv('BYTEPS_ENABLE_IPC', '1')

    os.environ['BYTEPS_USE_GDR_ALLGATHER'] = os.getenv('BYTEPS_USE_GDR_ALLGATHER', '0')

    # for ucx only
    os.environ['UCX_RDMA_CM_SOURCE_ADDRESS'] = os.environ['DMLC_NODE_HOST']
    os.environ['UCX_SOCKADDR_CM_ENABLE'] = os.getenv('UCX_SOCKADDR_CM_ENABLE', 'y')

    os.environ['PYTHONUNBUFFERED'] = '1'

    # core binding
    if is_a100_node():
        _set_core_affinity_a100()
    else:
        _set_core_affinity_v100()

    if not ProcessGroupBYTEPS.byteps_inited:
        import time
        start_scheduler_server()
        time.sleep(5)
        if os.environ['BYTEPS_LOCAL_RANK'] == "0":
            print("===========================================================")
            os.system('ip a')
            print("===========================================================")
            os.system('nvidia-smi topo -m')
            print("===========================================================")
        if os.getenv('BYTEPS_DEBUG_DUMP_ENVARS'):
            local_rank = int(os.environ['BYTEPS_LOCAL_RANK'])
            time.sleep(local_rank)
            _dump_byteps_relevant_env_vars(desc=f"byteps_worker rank: {local_rank}")

        bps_init()
        ProcessGroupBYTEPS.byteps_inited = True

    pg = ProcessGroupBYTEPS(store=prefix_store, rank=rank, size=world_size,
            timeout=timeout)
    print("Created process group using the BytePS backend, rank=", rank)
    return pg
