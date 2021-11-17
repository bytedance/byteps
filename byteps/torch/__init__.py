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

from byteps.torch.compression import Compression
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import push_pull
from byteps.torch.ops import batched_fuse_, batched_unfuse_, batched_zero_
from byteps.torch.ops import byteps_torch_set_num_grads
from byteps.torch.ops import poll, synchronize, declare
from byteps.torch.ops import init, shutdown, suspend, resume
from byteps.torch.ops import size, local_size, rank, local_rank
from byteps.torch.ops import send_async, recv_async
from byteps.torch.optimizer import _DistributedOptimizer
from byteps.torch.functions import broadcast_parameters, broadcast_optimizer_state, broadcast_object
from byteps.torch.sync_batch_norm import SyncBatchNorm

def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1, staleness=0,
                         pipesgd_warmup_iter=0,
                         model=None, **kwargs):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an push_pull to
    average gradient values before applying gradients to model weights.
    push_pull operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all push_pull operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces push_pull operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:

    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          push_pull operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during push_pull to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
        staleness: Number of controlled gradients staleness if pipelined SGD is enabled. 
                   This allows optimizer using stale gradients to update parameters. Defaults 
                   to not using pipelined SGD, i.e., staleness=0. If set to 1, the parameter
                   update is delayed by 1 step. Reference: https://arxiv.org/abs/1811.03619
        pipesgd_warmup_iter: Number of warmup steps for pipesgd, during which pipesgd staleness
                   is fixed at 0.
        model: The model being trained. Passing the model in enables tensor
               fusion.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an push_pull implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               compression, backward_passes_per_step,
               staleness, pipesgd_warmup_iter=pipesgd_warmup_iter,
               model=model, **kwargs)
