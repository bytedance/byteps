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

import torch

from .ProcessGroupBYTEPS import _new_byteps_process_group_hook, _set_default_env
from .ProcessGroupBYTEPS import ProcessGroupBYTEPS

from torch.distributed.constants import default_pg_timeout
from torch.distributed import Backend as Backend
from torch.distributed import broadcast_object_list as torch_broadcast_object_list
from torch.distributed import all_gather_object as torch_all_gather_object
from torch.distributed import Store
from . import init_process_group as torch_init_process_group
from .rendezvous import register_rendezvous_handler
import logging

logger = logging.getLogger(__name__)

Backend.register_backend(
        name="byteps",
        func=_new_byteps_process_group_hook)

def byteps_init_process_group(
    backend,
    init_method=None,
    timeout=default_pg_timeout,
    world_size=-1,
    rank=-1,
    store=None,
    group_name="",
    pg_options=None,
):
    """
    Initializes the default distributed process group, and this will also
    initialize the distributed package.

    This is a modified version of
    torch/distributed/distributed_c10d.py::init_process_group, this function is
    only supposed to be called with ``backend="nccl"`` or
    ``backend=Backend.NCCL``, and the BytePS backend is used in place of nccl.
    Only the following arguments are meaningful:

    Args:
        backend (str or Backend): The backend to use.This field should be given
        as a lowercase string: ``"nccl"``, it can also be accessed via
            :class:`Backend` attributes: ``Backend.NCCL``.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).
                              Required if ``store`` is specified.


    """
    try:
        import byteps.torch as bps
    except (ImportError, ModuleNotFoundError):
        import warnings
        msg = "BytePS is not found, the native torch.distrbuted package " \
              "and the native torch.nn.parallel.DistributedDataParallel " \
              "class will be used."
        warnings.warn(msg)
        torch_init_process_group(*args, **kwargs)
        return

    backend = Backend(backend)
    if backend == Backend.NCCL:
        backend = "byteps"
        import os
        local_size = torch.cuda.device_count()
        _set_default_env('WORLD_SIZE', world_size)
        _set_default_env('LOCAL_WORLD_SIZE', local_size)
        _set_default_env('RANK', rank)
        rank = int(os.getenv('RANK'))
        _set_default_env('LOCAL_RANK', str(rank % local_size))
        os.environ['DMLC_WORKER_ID'] = str(rank // local_size)
        _set_default_env('DMLC_PS_ROOT_PORT', 59000)

    torch_init_process_group(backend=backend, init_method="byteps://",
        world_size=world_size, rank=rank, group_name=group_name)

init_process_group = byteps_init_process_group

def byteps_broadcast_object_list(object_list, src=0, group=None, device=None):
    if torch.cuda.is_available():
        current_device = torch.device("cuda", torch.cuda.current_device())
    else:
        current_device = torch.device("cpu")
    if device is not None and device != current_device:
        raise ValueError("device must be the current GPU for the BytePS backend")
    torch_broadcast_object_list(object_list=object_list, src=src, group=group,
                                device=current_device)

broadcast_object_list = byteps_broadcast_object_list

def byteps_all_gather_object(object_list, obj, group=None):

    if torch.cuda.is_available():
        real_ProcessGroupNCCL = torch.distributed.distributed_c10d.ProcessGroupNCCL
        torch.distributed.distributed_c10d.ProcessGroupNCCL = ProcessGroupBYTEPS

    torch_all_gather_object(object_list, obj, group)

    if torch.cuda.is_available():
        torch.distributed.distributed_c10d.ProcessGroupNCCL = real_ProcessGroupNCCL

all_gather_object = byteps_all_gather_object

class BytepsStore(Store):
    def __init__(self, rank=-1, world_size=-1):
        super().__init__()
        self._rank = rank
        self._world_size = world_size

    def set(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        return self._world_size

    def delete_key(self, *args, **kwargs):
        pass

    def num_keys(self, *args, **kwargs):
        pass

    def set_timeout(self, *args, **kwargs):
        pass

    def wait(self, *args, **kwargs):
        pass

def _byteps_rendezvous_handler(*args, **kwargs):
    def _error(msg):
        return ValueError("Error initializing torch.distributed using byteps:// rendezvous: " + msg)

    import os
    query = os.environ
    if "RANK" not in query:
        raise _error("RANK envar missing")
    if "WORLD_SIZE" not in query:
        raise _error("WORLD_SIZE envar missing")

    rank = int(query["RANK"])
    world_size = int(query["WORLD_SIZE"])

    store = BytepsStore(rank, world_size)
    logger.info(
        f"Rank {rank}: Using the BytepsStore"
    )

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using byteps:// method")

register_rendezvous_handler("byteps", _byteps_rendezvous_handler)
