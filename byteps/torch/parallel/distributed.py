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
from byteps.torch.grad_fusion import _GradFusion, _find_tensors, find_parameters
import collections

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
            compression=Compression.none,
            bucket_cap_mb=25,
            find_unused_parameters=False,
            *args, **kwargs
            ):
        super(DistributedDataParallel, self).__init__()

        if devices_ids is None:
            self.device_ids = None
        else:
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
        ##################
        self._pipesgd_warmup_iter = int(os.getenv('BYTEPS_PIPESGD_WARMUP_ITER', '0'))
        self._staleness = int(os.getenv('BYTEPS_STALENESS', '0'))
        self.state = {'byteps_niter': 0}
        self.backward_passes_per_step = 1
        self._push_pull_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        # for pipesgd
        self._stale_handles = collections.defaultdict(dict)
        self._stale_handles[-1] = {}
        self._stale_push_pull_delay = collections.defaultdict(dict)
        self._stale_push_pull_delay = {self._get_param_name(v): self.backward_passes_per_step
                                       for _, v in sorted(named_parameters)}
        # whether it is the initial optimizer.step()
        self.state['byteps_skipped_init_step'] = False
        # a reference of self._niter stored in states
        # self.state['byteps_niter'] = 0
        # store the scale factor from amp if fp16 dynamic scaling is present
        self.state['byteps_stale_scale'] = 1
        # checkpoint the materialized gradient before torch.save() is called.
        # the ckpt grad will be used in three cases: the iteration after checkpoint,
        # the 1-st iteration after resuming training from ckpt, and the first iteration
        # that pipesgd takes effect.
        self.state['byteps_stale_grad'] = {}

        ##################
        self._enable_tensor_fusion = False
        self._grad_fusion = None
        named_params = self.module.named_parameters()
        self._trainable_params = [p for _, p in named_params if p.requires_grad]
        named_params = self.module.named_parameters()
        self._num_grads = sum(p.requires_grad for _, p in named_params)
        if size() > 1:
            if bucket_cap_mb > 0:
                self._enable_tensor_fusion = True
                os.environ['BYTEPS_BUCKET_SIZE_BYTES'] = str(bucket_cap_mb * 1024 * 1024)
                self._grad_fusion = _GradFusion(module, self)
                self._num_grads = self._grad_fusion.size()
            elif find_unused_parameters:
                self.register_forward_hook(self._model_post_fwd_hook)
            self._register_hooks()
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

        print("Using the BytePS DistributedDataParallel Module")
        self._step = -1
        self._empty_cache = os.getenv("BYTEPS_TORCH_CUDA_EMPTY_CACHE") in ["1"]

    def _get_param_name(self, p):
        """Get the name of a parameter."""
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        return name

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
        if self._empty_cache:
            torch.cuda.empty_cache()
        self._step += 1
        num_handles = len(self._handles)
        assert num_handles == 0, f'step {self._step} num_handles is {num_handles}'
        torch.cuda.nvtx.range_push("byteps.DistributedDataParallel.forward")
        if self.require_forward_param_sync:
            self._sync_params()
        output = self.module(*inputs, **kwargs)
        torch.cuda.nvtx.range_pop()
        return output

    def _sync_params(self):
        torch.cuda.nvtx.range_push("_sync_params")
        with torch.no_grad():
            # sync module buffers
            if self.broadcast_buffers and len(self.modules_buffers[0]) > 0:
                # Synchronize buffers across processes.
                # The process with rank 0 is considered the authoritative copy.
                bps.torch.broadcast_parameters(list(self.module.named_buffers()), root_rank=0)
        torch.cuda.nvtx.range_pop()

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
        if self._enable_tensor_fusion and size() > 1:
            return self._grad_fusion._make_hook(p)

        def hook(*ignore):
            if self._require_backward_grad_sync:
                handle, ctx = None, None
                handle, ctx, grad_count = self._push_pull_grad_group_sync(p, num_grads)
                self._handles[p] = (handle, ctx)
                # sync if we have processed all gradients
                if grad_count == self._num_grads:
                    self._synchronize()
        return hook

    def _model_post_fwd_hook(self, module, _inputs, outputs):
        ''' post model forward hook. '''
        if not module.training:
            return

        out_tensors = list(_find_tensors(outputs))
        params_in_graph, _ = find_parameters(out_tensors)
        self._num_grads = len(set(params_in_graph) & set(self._trainable_params))
        byteps_torch_set_num_grads(self._num_grads)

    def _normal_synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            if type(p.grad) == type(None):
                continue

            assert False, "This should never be reached."
            handle, ctx, grad_count = self._push_pull_grad_group_sync(p, self._num_grads)
            self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx, grad_count = self._push_pull_grad_group_sync(p)
                self._handles[p] = (handle, ctx)
        for p, (handle, ctx) in self._handles.items():
            output = synchronize(handle)
            if not self._enable_async:
                if type(p.grad) == type(None):
                    assert False, "This should never be reached."
                    continue
                p.grad.set_(self._compression.decompress(output, ctx))
        self._handles.clear()

    def _stale_synchronize(self):
        """Synchronize the pushpull operations when pipesgd is enabled"""
        has_amp = hasattr(self, "_amp_stash")
        niter = self.state['byteps_niter']
        assert niter >= 0, niter
        loss_scale = _get_loss_scale()
        # if the loss scale increases at the current iteration
        # amp will rescale it back after synchronize(). so we need
        # to adjust the gradient from the previous step accordingly
        if niter > self._pipesgd_warmup_iter:
            prev_loss_scale = self.state['byteps_stale_scale']
        else:
            prev_loss_scale = loss_scale
        grad_ratio = loss_scale / prev_loss_scale

        # materialzed grad tensors are not available. obtain them from handles
        stale_grad_state = self.state['byteps_stale_grad']
        if not stale_grad_state:
            if niter <= self._pipesgd_warmup_iter:
                if niter == self._pipesgd_warmup_iter:
                    print(f'BytePS pipeSGD: started pipeline at iter {niter}', flush=True)
                for name, (p, handle, ctx) in self._stale_handles[niter].items():
                    assert handle is not None, name
                    assert not p.grad.is_sparse, "sparse gradient is not supported"
                    output = synchronize(handle)
                    tmp = self._compression.decompress(output, ctx)
                    # sync SGD duration warmup
                    if niter < self._pipesgd_warmup_iter:
                        p.grad.data = tmp
                    else:
                        stale_grad_state[name] = tmp
                        p.grad.copy_(tmp)
            else:
                for name, (p, handle, ctx) in self._stale_handles[niter-1].items():
                    assert handle is not None
                    assert not p.grad.is_sparse, "sparse gradient is not supported"
                    output = synchronize(handle)
                    prev_grad = self._compression.decompress(output, ctx)
                    with torch.no_grad():
                        # if the loss scale increases at the current iteration
                        # amp will rescale it back after synchronize(). so we need
                        # to adjust the gradient from the previous step accordingly
                        if grad_ratio != 1.0:
                            prev_grad.mul_(grad_ratio)
                        p.grad.copy_(prev_grad)
            if (niter - 1) in self._stale_handles:
                del self._stale_handles[niter - 1]
        else:
            # grad tensors alread materialized
            for p in self._requires_update:
                assert not p.grad.is_sparse, "sparse gradient is not supported"
                name = self._get_param_name(p)
                if name in stale_grad_state:
                    prev_grad = stale_grad_state[name]
                    with torch.no_grad():
                        if grad_ratio != 1.0:
                            prev_grad.mul_(grad_ratio)
                        p.grad.copy_(prev_grad)
            self.state['byteps_stale_grad'] = {}

        # update states
        for name in self._stale_push_pull_delay:
            self._stale_push_pull_delay[name] = self.backward_passes_per_step
        self.state['byteps_stale_scale'] = loss_scale
        self.state['byteps_niter'] += 1

    def _synchronize(self):
        if self._staleness:
            self._stale_synchronize()
        else:
            self._normal_synchronize()
