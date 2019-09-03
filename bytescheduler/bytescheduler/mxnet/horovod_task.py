from __future__ import absolute_import

import os
from ctypes import CDLL, RTLD_GLOBAL, CFUNCTYPE, byref, c_void_p

from mxnet.base import check_call, NDArrayHandle
from mxnet.ndarray import NDArray
from ..common import get_ext_suffix
from ..common.bytetask import ByteTask


# Load c_lib.so
dll_path = os.path.join(os.path.dirname(__file__),
                        'c_lib' + get_ext_suffix())
BYTESCHEDULER_LIB = CDLL(dll_path, RTLD_GLOBAL)
callback_t = CFUNCTYPE(None, c_void_p)


class HorovodTask(ByteTask):

    def _post_communication(self, tensor):
        """Start allreduce a tensor
        Args:
            tensor: a list of tensor to be allreduced.
        """
        self._comm._do_allreduce(self.name, tensor)

    def _do(self):
        """Let the start OP complete so that the real comm OP can run."""
        if hasattr(self, "_on_complete"):
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                c_void_p(self._on_complete)))
        return

    def _prepare(self):
        """Post start barrier OP, start OP, comm OP, end OP and end barrier OP to MXNet engine. The function of each
        kind of OP is explained below.
        start barrier OP: barrier the start of a parent ByteTask, used to maintain original dependency.
        start OP: It notifies Core about task readiness. It is also used to delay the start of a child ByteTask.
        comm OP: the OP that does real communication, e.g., push, pull, allreduce.
        end OP: an OP that runs after a child ByteTask is finished. It notifies Core about the task completion.
        end barrier OP: an OP that runs after the parent ByteTask is finished, used to maintain original dependency.
        """
        if self.parent is None:
            real = self._tensor.handle
            avatar = NDArrayHandle()
            check_call(BYTESCHEDULER_LIB.bytescheduler_get_ndarray_avatar(
                real, byref(avatar)))
            self._avatar = NDArray(avatar)
            avatar = self._avatar.handle
        else:
            real = self.parent._tensor.handle
            avatar = self._tensor.handle

        self._post_start_barrier(avatar, real)
        self._post_start_op(avatar)

        # Post real op
        if self.parent is None:
            self._post_communication(self._avatar)
            
        else:
            self._post_communication(self._tensor)

        self._post_end_op(avatar)
        self._post_end_barrier(avatar, real)

    def _post_start_barrier(self, avatar, real):
        """The start barrier is for keeping the original dependency. It does not need any callback."""
        if self.parent is None:
            barrier_tensors_out = (NDArrayHandle * 2)(*[real, avatar])
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_out, 0,
                barrier_tensors_out, 2,
                10000000-self.priority))
        else:
            if hasattr(self.parent, "_posted_start_barrier"):
                return
            self.parent._posted_start_barrier = True
            children_tensors = []
            for child in self.parent.children:
                children_tensors.append(child._tensor.handle)
            barrier_tensors_out = (NDArrayHandle * (len(children_tensors) + 1))(
                *(children_tensors + [real]))
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_out, 0,
                barrier_tensors_out, len(children_tensors) + 1,
                10000000-self.priority))

    def _post_start_op(self, avatar):
        """The start op is only for notifying the Core about task ready. It does not add any dependency to the
        original DAG."""
        def start_callback(on_complete):
            if self._immediate:
                # Call on_complete directly if it is an immediate task.
                check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                    c_void_p(on_complete)))
                self._logger.debug("task {} {} is ready.".format(self.op, self.name))
                return
            self._on_complete = on_complete
            self.notify_ready()

        # Avoid garbage collection
        self._mxnet_start_callback = callback_t(start_callback)

        # Post start op
        tensor_out = (NDArrayHandle * 1)(*[avatar])
        check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
            tensor_out, 0, tensor_out, 1, self._mxnet_start_callback, 1000000-self.priority))

    def _post_end_op(self, avatar):
        """The end op is only for notifying the Core about task finishing. It does not add any dependency to the
        original DAG."""
        def end_callback(on_complete):
            # Call on_complete directly
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                c_void_p(on_complete)))
            self.notify_finish()

        # Avoid garbage collection
        self._mxnet_end_callback = callback_t(end_callback)

        # Post end op
        tensor_out = (NDArrayHandle * 1)(*[avatar])
        check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
            tensor_out, 0, tensor_out, 1, self._mxnet_end_callback, 1000000-self.priority))

    def _post_end_barrier(self, avatar, real):
        """The end barrier is for keeping the original dependency. It does not need any callback."""
        if self.parent is None:
            barrier_tensors_in = (NDArrayHandle * 1)(*[avatar])
            barrier_tensors_out = (NDArrayHandle * 1)(*[real])
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_in, 1,
                barrier_tensors_out, 1,
                10000000-self.priority))
        else:
            if not hasattr(self.parent, "_children_tensors"):
                self.parent._children_tensors = [avatar]
            else:
                self.parent._children_tensors.append(avatar)
            if len(self.parent._children_tensors) == len(self.parent.children):
                barrier_tensors_in = (NDArrayHandle * len(self.parent.children))(
                    *self.parent._children_tensors)
                barrier_tensors_out = (NDArrayHandle * 1)(*[real])
                check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                    barrier_tensors_in, len(self.parent.children),
                    barrier_tensors_out, 1,
                    10000000-self.priority))

    def _tensor_size(self):
        return self._tensor.size

    def _partition_tensor(self, size):
        """Zero-copy implementation.
        Note: ndarray works for up to ~4 billion parameters.

        Below 2 lines are buggy with horovod -- causing bad performance.
            tmp = self._tensor.reshape(-1, 1)
            avatar = tmp[start:end]
        """
        number = (self._tensor.size - 1) // size + 1
        if number > self._tensor.shape[0]:
            self._logger.warning(
                "The number of tensor rows (with shape {}) is smaller than partition number {}.".format(
                    self._tensor.shape, number))
            number = self._tensor.shape[0]
        num_per_partition = self._tensor.shape[0] // number
        partitions_with_extra = self._tensor.shape[0] % number

        partitions = []
        start = 0
        end = num_per_partition
        for i in range(number):
            handle = NDArrayHandle()
            check_call(BYTESCHEDULER_LIB.bytescheduler_get_ndarray_avatar(
                self._tensor.handle, byref(handle)))
            avatar = NDArray(handle)[start:end]
            partitions.append(avatar)
            start = end
            end += num_per_partition
            if i >= number - partitions_with_extra - 1:
                end += 1
        return partitions

    def _immediate_do(self):
        self._prepare()
