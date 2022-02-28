from .distributed import DistributedDataParallel as DistributedDataParallel
from .sync_batchnorm import SyncBatchNorm

def convert_syncbn_model(module, process_group=None, channel_last=False):
    '''
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`apex.parallel.SyncBatchNorm`.
    All ``torch.nn.BatchNorm*N*d`` wrap around
    ``torch.nn.modules.batchnorm._BatchNorm``, so this function lets you easily switch
    to use sync BN.
    Args:
        module (torch.nn.Module): input module
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> import apex
        >>> sync_bn_model = apex.parallel.convert_syncbn_model(model)
    '''
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, process_group, channel_last=channel_last)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_syncbn_model(child,
                                                  process_group=process_group,
                                                  channel_last=channel_last))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod

__all__ = ['DistributedDataParallel', 'SyncBatchNorm', 'convert_syncbn_model']
