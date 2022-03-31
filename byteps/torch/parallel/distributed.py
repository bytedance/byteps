import torch
from torch.nn.modules import Module
from byteps.torch.ops import push_pull_group_sync_inplace as byteps_push_pull_group
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import poll, synchronize, declare, byteps_torch_set_num_grads
from byteps.torch.ops import size, local_size, rank, local_rank
from contextlib import contextmanager
import byteps as bps
from byteps.torch.compression import Compression
from torch.cuda._utils import _get_device_index
import os

class DistributedDataParallel(Module):
    r"""Implements distributed data parallelism that is based on
    byteps push-pull.

    This container parallelizes the application of the given module by splitting
    the input across multiple devices, and each device handles a portion of the
    input. During the backwards pass, gradients from each node are averaged.

    ``DistributedDataParallel`` can be used in the following way:

    Single-Process Single-GPU

    This is currently the only way to use ``DistributedDataParallel``, with
    multiple processes, each of which operates on a single GPU.

    Here is how to use it: on each host with N GPUs, you should spawn up N
    processes, while ensuring that each process individually works on a single
    GPU from 0 to N-1. Therefore, it is your job to ensure that your training
    script operates on a single given GPU by calling:

        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> model = DistributedDataParallel(model, device_ids=[i])

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. warning::
        This module works only with the ``device_ids`` containing one entry.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) is a distributed synchronization
        point. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all buffers and gradients are dense.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    .. warning::
        You should never try to change your model's parameters after wrapping
        up your model with DistributedDataParallel. In other words, when
        wrapping up your model with DistributedDataParallel, the constructor of
        DistributedDataParallel will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters after
        the DistributedDataParallel construction, this is not supported and
        unexpected behaviors can happen, since some parameters' gradient
        reduction functions might not get called.

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in rank 0 to all
        other replicas in the system in every iteration.

    .. note::
        Some models have branches, part of the model is skipped during the
        forward pass. In that case it's required to call the
        DistributedDataParallel.synchronize() after loss.backward(), e.g:

            >>> model = DistributedDataParallel(model, device_ids=[i])
            >>> output = model(data)
            >>> loss = F.nll_loss(output, target)
            >>> loss.backward()
            >>> model.synchronize()
            >>> optimizer.step()

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices. This should
                   contain only one entry. The `module` replica is placed on
                   ``device_ids[0]``.
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
                          the module at beginning of the forward function.
                          (default: ``True``)

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> net = torch.nn.DistributedDataParallel(model, device_ids=[2])
    """
    def __init__(self, module, device_ids=None,
            broadcast_buffers=True,
            compression=Compression.none
            ):
        super(DistributedDataParallel, self).__init__()

        assert device_ids and len(device_ids) == 1, (
                "DistributedDataParallel device_ids contain exactlyone entry,"
                " but got {}.").format(device_ids)
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        self.require_forward_param_sync = broadcast_buffers
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._num_grads = 1

        self.modules_buffers = [list(self.module.buffers())]
        self._compression = compression
        self._enable_async = False
        self._require_backward_grad_sync = True
        named_parameters = self.module.named_parameters()
        named_parameters = list(named_parameters)
        if len(named_parameters) > 0:
            if isinstance(named_parameters[0][1], torch.Tensor):
                if any([not isinstance(p, torch.Tensor) for name, p in named_parameters]):
                    raise ValueError('named_parameters should consistently be a sequence of '
                                     'tuples (name, torch.Tensor)')
                self._is_tensor_instance = True
                # there is an issue when using torch.Tensor as key, so use its hash instead
                # https://github.com/pytorch/pytorch/issues/7733
                self._parameter_names = {v.__hash__(): k for k, v
                                         in sorted(named_parameters)}
                self._tensor_list = [tensor for name, tensor in named_parameters]
            else:
                self._is_tensor_instance = False
                self._parameter_names = {v: k for k, v
                                         in sorted(named_parameters)}
        else:
            self._is_tensor_instance = False
            self._parameter_names = {v: 'push_pull.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        if size() > 1:
            self._register_hooks()
            named_params = self.module.named_parameters()
            self._num_grads = sum(p.requires_grad for _, p in named_params)
            byteps_torch_set_num_grads(self._num_grads)

        # declare tensors
        for name in sorted(self._parameter_names.values()):
            declare("Gradient."+name)
        # We use two loops for load-balancing
        for name in sorted(self._parameter_names.values()):
            declare("Parameter."+name)

        # broadcast model state
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            bps.torch.broadcast_parameters(self.module.state_dict(), root_rank=0)

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> ddp = byteps.torch.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            ...   for input in inputs:
            ...     ddp(input).backward()  # no synchronization, accumulate grads
            ... ddp(another_input).backward()  # synchronize grads
        """
        if self._enable_async:
            raise AssertionError("no_sync cannot be used in async training")
        old_require_backward_grad_sync = self._require_backward_grad_sync
        self._require_backward_grad_sync = False
        try:
            yield
        finally:
            self._require_backward_grad_sync = old_require_backward_grad_sync

    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()
        return self.module(*inputs, **kwargs)

    def _sync_params(self):
        with torch.no_grad():
            # sync module buffers
            if self.broadcast_buffers and len(self.modules_buffers[0]) > 0:
                # Synchronize buffers across processes.
                # The process with rank 0 is considered the authoritative copy.
                bps.torch.broadcast_parameters(list(self.module.named_buffers()), root_rank=0)

    def _register_hooks(self):
        for _, p in self.module.named_parameters():
            if p.requires_grad:
                p.grad = p.data.new(p.size()).zero_()
                self._requires_update.add(p)
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(p, self._num_grads))
                self._grad_accs.append(grad_acc)

    def _push_pull_grad_group_sync(self, p, num_grads_):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        if self._enable_async:
            # the real handle will be created in step()
            handle, ctx = None, None
        else:
            tensor = p.grad
            tensor_compressed, ctx = self._compression.compress(tensor)
            handle, grad_count = byteps_push_pull_group(tensor_compressed, average=True,
                    name="Gradient."+name)
        return handle, ctx, grad_count

    def _push_pull_grad_async(self, p):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        if self._enable_async:
            # the real handle will be created in step()
            handle, ctx = None, None
        else:
            tensor = p.grad
            tensor_compressed, ctx = self._compression.compress(tensor)
            handle = byteps_push_pull(tensor_compressed, average=True, name="Gradient."+name)
        return handle, ctx

    def _make_hook(self, p, num_grads):
        def hook(*ignore):
            if self._require_backward_grad_sync:
                handle, ctx = None, None
                handle, ctx, grad_count = self._push_pull_grad_group_sync(p, num_grads)
                self._handles[p] = (handle, ctx)
                # sync if we have processed all gradients
                if grad_count == self._num_grads:
                    self.synchronize()
        return hook

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx, grad_count = self._push_pull_grad_group_sync(p, self._num_grads)
            self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx, grad_count = self._push_pull_grad_group_sync(p)
                self._handles[p] = (handle, ctx)
        for p, (handle, _) in self._handles.items():
            output = synchronize(handle)
            if not self._enable_async:
                p.grad.set_(self._compression.decompress(output, ctx))
        self._handles.clear()
