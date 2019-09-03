from __future__ import absolute_import
import horovod.mxnet as hvd
from .horovod_task import HorovodTask
from ..common.bytecore import core


class ScheduledOptimizer(hvd.DistributedOptimizer):
    """An optimizer that wraps a hvd.DistributedOptimizer, intercepting allreduce operations and wrap as tasks.

    Usage example:
    ```
    import bytescheduler.mxnet.horovod as bsc
    bsc.init()
    # opt is an MXNet optimizer created by `mx.optimizer.create()`.
    opt = hvd.DistributedOptimizer(opt)
    ```
    """

    def __init__(self, optimizer):
        """Construct a new ScheduledOptimizer, which uses horovod optimizer under the hood for averaging gradients
        across all the Horovod ranks.

        Args:
            optimizer: Optimizer to use for computing and averaging gradients and applying updates.
        """
        self._optimizer = optimizer
        self._immediate = False

        # Let rank 0 decide the communication order
        self._rank = hvd.rank()
        if self._rank != 0:
            self._immediate = True

        self._first_key = None
        self._step = 0

        core.start(rank=self._rank, arch="allreduce")

    def _post_task(self, index, weight, grad, state, update_fn):
        """Post each allreduce operation as a ByteTask to Core.

        Index is used as priority. We assume the closer to input layer, the smaller the index is. So smaller index
        indicates higher priority.
        """
        # Count training step
        if not self._first_key:
            self._first_key = index
        if self._first_key == index:
            self._step += 1

        # The post function will maintain correct dependency so that update_fn function is executed after allreduce.
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                task = HorovodTask(
                    str(index[i]),
                    grad[i],
                    "allreduce",
                    priority=index[i],
                    comm=self,
                    immediate=self._immediate,
                    step=self._step,
                    rank=self._rank)
                core.post(task)
        else:
            task = HorovodTask(
                str(index),
                grad,
                "allreduce",
                priority=index,
                comm=self,
                immediate=self._immediate,
                step=self._step,
                rank=self._rank)
            core.post(task)

        update_fn(index, weight, grad, state)

    def update(self, index, weight, grad, state):
        """Override the default one"""
        self._post_task(index, weight, grad, state, self._optimizer.update)

    def update_multi_precision(self, index, weight, grad, state):
        """Override the default one"""
        self._post_task(index, weight, grad, state, self._optimizer.update_multi_precision)


def init():
    """Override horovod's optimizer"""
    hvd.DistributedOptimizer = ScheduledOptimizer
