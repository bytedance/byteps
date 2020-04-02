import torch
from torch.nn.modules import Module
from byteps.torch.ops import push_pull_group_sync_inplace as byteps_push_pull
from byteps.torch.ops import poll, synchronize, declare, byteps_torch_set_num_grads
from contextlib import contextmanager
from byteps.torch.ops import size, local_size, rank, local_rank
import byteps as bps
from byteps.torch.compression import Compression

class DistributedDataParallel(Module):
    r"""Implements distributed data parallelism that is based on
    ``torch.distributed`` package at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. During the backwards
    pass, gradients from each node are averaged.

    The batch size should be larger than the number of GPUs used locally.

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-dataparallel-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`.

    ``DistributedDataParallel`` can be used in the following two ways:

    (1) Single-Process Multi-GPU

    In this case, a single process will be
    spawned on each host/node and each process will operate on all the GPUs
    of the node where it's running. To use ``DistributedDataParallel`` in
    this way, you can simply construct the model as the following:

        >>> torch.distributed.init_process_group(backend="nccl")
        >>> model = DistributedDataParallel(model) # device_ids will include all GPU devices by default

    (2) Multi-Process Single-GPU

    This is the highly recommended way to use ``DistributedDataParallel``, with
    multiple processes, each of which operates on a single GPU. This is
    currently the fastest approach to do data parallel training using PyTorch
    and applies to both single-node(multi-GPU) and multi-node data
    parallel training. It is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    Here is how to use it: on each host with N GPUs, you should spawn up N
    processes, while ensuring that each process individually works on a single GPU
    from 0 to N-1. Therefore, it is your job to ensure that your training script
    operates on a single given GPU by calling:

        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``

    .. note:: ``nccl`` backend is currently the fastest and
        highly recommended backend to be used with Multi-Process Single-GPU
        distributed training and this applies to both single-node and multi-node
        distributed training

    .. note:: This module also supports mixed-precision distributed training.
        This means that your model can have different types of parameters such
        as mixed types of fp16 and fp32, the gradient reduction on these
        mixed types of parameters will just work fine.
        Also note that ``nccl`` backend is currently the fastest and highly
        recommended backend for fp16/fp32 mixed-precision training.

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. warning::
        This module works only with the ``gloo`` and ``nccl`` backends.

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
        This module assumes all parameters are registered in the model of each
        distributed processes are in the same order. The module itself will
        conduct gradient all-reduction following the reverse order of the
        registered parameters of the model. In other words, it is users'
        responsibility to ensure that each distributed process has the exact
        same model and thus the exact same parameter registration order.

    .. warning::
        This module assumes all buffers and gradients are dense.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::

        If you plan on using this module with a ``nccl`` backend or a ``gloo``
        backend (that uses Infiniband), together with a DataLoader that uses
        multiple workers, please change the multiprocessing start method to
        ``forkserver`` (Python 3 only) or ``spawn``. Unfortunately
        Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will
        likely experience deadlocks if you don't change this setting.

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
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices. This should
                   only be provided when the input module resides on a single
                   CUDA device. For single-device modules, the ``i``th
                   :attr:`module` replica is placed on ``device_ids[i]``. For
                   multi-device modules and CPU modules, device_ids must be None
                   or an empty list, and input data for the forward pass must be
                   placed on the correct device. (default: all devices for
                   single-device modules)
        output_device (int or torch.device): device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be None, and the module itself
                      dictates the output location. (default: device_ids[0] for
                      single-device modules)
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
                          the module at beginning of the forward function.
                          (default: ``True``)
        process_group: the process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by ```torch.distributed.init_process_group```,
                       will be used. (default: ``None``)
        bucket_cap_mb: DistributedDataParallel will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in MegaBytes (MB)
                       (default: 25)
        find_unused_parameters (bool): Traverse the autograd graph of all tensors
                                       contained in the return value of the wrapped
                                       module's ``forward`` function.
                                       Parameters that don't receive gradients as
                                       part of this graph are preemptively marked
                                       as being ready to be reduced. Note that all
                                       ``forward`` outputs that are derived from
                                       module parameters must participate in
                                       calculating loss and later the gradient
                                       computation. If they don't, this wrapper will
                                       hang waiting for autograd to produce gradients
                                       for those parameters. Any outputs derived from
                                       module parameters that are otherwise unused can
                                       be detached from the autograd graph using
                                       ``torch.Tensor.detach``. (default: ``False``)
        check_reduction: when setting to ``True``, it enables DistributedDataParallel
                         to automatically check if the previous iteration's
                         backward reductions were successfully issued at the
                         beginning of every iteration's forward function.
                         You normally don't need this option enabled unless you
                         are observing weird behaviors such as different ranks
                         are getting different gradients, which should not
                         happen if DistributedDataParallel is correctly used.
                         (default: ``False``)

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model, pg)
    """
    def __init__(self, module, device_ids=None,
            output_device=None, dim=0, broadcast_buffers=True,
            process_group=None, bucket_cap_mb=25,
            find_unused_parameters=False,
            check_reduction=False,
            compression=Compression.none
            ):
        super(DistributedDataParallel, self).__init__()

        self.module = module
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._num_grads = 1
        if size() > 1:
            self._register_hooks()
            named_params = self.module.named_parameters()
            self._num_grads = sum(p.requires_grad for _, p in named_params)
            byteps_torch_set_num_grads(self._num_grads)

        self._compression = compression
        self._enable_async = False
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
        # declare tensors
        for name in sorted(self._parameter_names.values()):
            declare("Gradient."+name)
        # We use two loops for load-balancing
        for name in sorted(self._parameter_names.values()):
            declare("Parameter."+name)

        # broadcast parameters
#        bps.torch.broadcast_parameters(named_parameters, root_rank=0)
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            bps.torch.broadcast_parameters(self.module.state_dict(), root_rank=0)


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> ddp = torch.nn.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            ...   for input in inputs:
            ...     ddp(input).backward()  # no synchronization, accumulate grads
            ... ddp(another_input).backward()  # synchronize grads
        """
        pass

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
            # todo: need to make a new function which calls synchronize on the
            # last parameter
            handle = byteps_push_pull(tensor_compressed, average=True,
                    name="Gradient."+name)
        return handle, ctx

    def _make_hook(self, p, num_grads):
        def hook(*ignore):
            handle, ctx = None, None
            handle, ctx = self._push_pull_grad_group_sync(p, num_grads)
            self._handles[p] = (handle, ctx)
        return hook
