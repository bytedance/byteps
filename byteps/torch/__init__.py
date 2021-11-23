# Copyright 2019 Bytedance Inc. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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
from byteps.torch.compression import Compression
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import push_pull, allgather
from byteps.torch.ops import batched_fuse_, batched_unfuse_, batched_zero_
from byteps.torch.ops import byteps_torch_set_num_grads
from byteps.torch.ops import poll, synchronize, declare
from byteps.torch.ops import init, shutdown, suspend, resume
from byteps.torch.ops import size, local_size, rank, local_rank
from byteps.torch.ops import send_async, recv_async
from byteps.torch.optimizer import DistributedOptimizer
from byteps.torch.functions import broadcast_parameters, broadcast_optimizer_state, broadcast_object
from byteps.torch.sync_batch_norm import SyncBatchNorm
from . import parallel
