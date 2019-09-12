from __future__ import absolute_import

import os
from ctypes import CDLL, RTLD_GLOBAL, CFUNCTYPE, c_int, byref, c_void_p, c_char_p

from mxnet.base import check_call, NDArrayHandle
from mxnet.ndarray import NDArray
from mxnet.ndarray import zeros
from ..common import get_ext_suffix
from ..common.bytetask import ByteTask


# Load c_lib.so
dll_path = os.path.join(os.path.dirname(__file__),
                        'c_lib' + get_ext_suffix())
BYTESCHEDULER_LIB = CDLL(dll_path, RTLD_GLOBAL)
callback_t = CFUNCTYPE(None, c_void_p)

barrier_tensors = {}


def get_barrier_tensor(key):
    global barrier_tensors
    if key not in barrier_tensors:
        barrier_tensors[key] = zeros(1)
    return barrier_tensors[key]


class KVStoreTask(ByteTask):
    def _additional_init(self):
        if self.op == "push_pull":
            assert len(self._tensor) == 2
            self._push_tensor = self._tensor[0]
            self._pull_tensor = self._tensor[1]

            # unique key, assuming at most 10^6 tensors and each can be partitioned into at most 1000 partition
            self._barrier_key = str(self._partition_index + self.kwargs["key_index"]*1000 + 10**6)
            # used as a barrier for push_pull task
            self._barrier_tensor = get_barrier_tensor(self._barrier_key)
            self._barrier_op_name = "ByteSchedulerPush " + str(self.kwargs["key_index"])
        elif self.op == "init":
            # worker 0 needs to init barrier tensor on PS
            self._barrier_key = str(self._partition_index + self.kwargs["key_index"]*1000 + 10**6)
            self._barrier_tensor = get_barrier_tensor(self._barrier_key)
            self._barrier_op_name = "ByteSchedulerPush " + str(self.kwargs["key_index"])
            self._comm.init(self._barrier_key, self._barrier_tensor)

    def _post_communication(self, tensor):
        """Start send a tensor
        Args:
            tensor: a list of tensor to be init/push/pull.
        """
        if self.op == "init":
            self._comm.init(self.name, tensor)
        elif self.op == "push":
            self._comm.push(self.name, tensor, -self.priority)
        elif self.op == "pull":
            self._comm.pull(self.name, out=tensor, priority=-self.priority, ignore_sparse=self.kwargs["ignore_sparse"])
        elif self.op == "push_pull":
            assert len(tensor) == 2
            self._comm.push(self.name, tensor[0], -self.priority)
            # add an op to notify push completion
            self._push_completion_op_name = c_char_p(self.name.encode('ascii'))

            def push_completion_callback(on_complete):
                # Call on_complete directly
                check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                    c_void_p(on_complete)))
                # Called after push instead pull
                self.notify_finish()

            # Avoid garbage collection
            self._push_completion_callback = callback_t(push_completion_callback)
            self._comm.pull(self.name, out=tensor[1], priority=-self.priority, ignore_sparse=self.kwargs["ignore_sparse"])
        else:
            self._logger.error("ERROR: unexpected op type {}!".format(self.op))

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

        Below are several key data structures.

        self._tensor: a list of NDArrays of the same key of all devices. If push_pull, self._tensor includes push list
        and pull list of NDArrays of all devices.
        real: the original handle list of self._tensor, used for keep dependency.
        avatar: a new handle list of self._tensor.
        """
        if self.parent is None:
            if self.op == "push_pull":
                push_real = [t.handle for t in self._push_tensor] if isinstance(self._push_tensor, (tuple, list)) else [self._push_tensor.handle]
                pull_real = [t.handle for t in self._pull_tensor] if isinstance(self._pull_tensor, (tuple, list)) else [self._pull_tensor.handle]
                assert len(push_real) == len(pull_real)
                real = push_real + pull_real
            else:
                real = [t.handle for t in self._tensor] if isinstance(self._tensor, (tuple, list)) else [self._tensor.handle]
            avatar = []
            for h in real:
                avatar_h = NDArrayHandle()
                check_call(BYTESCHEDULER_LIB.bytescheduler_get_ndarray_avatar(
                    h, byref(avatar_h)))
                avatar.append(avatar_h)
            if self.op == "push_pull":
                # push avatar and pull avatar NDArrays
                self._avatar = [[NDArray(_) for _ in avatar[:int(len(avatar)/2)]], [NDArray(_) for _ in avatar[int(len(avatar)/2):]]]
                avatar = [_.handle for _ in self._avatar[0]] + [_.handle for _ in self._avatar[1]]
            else:
                self._avatar = [NDArray(_) for _ in avatar]
                avatar = [_.handle for _ in self._avatar]
        else:
            if self.op == "push_pull":
                push_real = [t.handle for t in self.parent._push_tensor] if isinstance(self.parent._push_tensor, (tuple, list)) else [self.parent._push_tensor.handle]
                pull_real = [t.handle for t in self.parent._pull_tensor] if isinstance(self.parent._pull_tensor, (tuple, list)) else [self.parent._pull_tensor.handle]
                real = push_real + pull_real
                push_avatar = [t.handle for t in self._push_tensor] if isinstance(self._push_tensor, (tuple, list)) else [self._push_tensor.handle]
                pull_avatar = [t.handle for t in self._pull_tensor] if isinstance(self._pull_tensor, (tuple, list)) else [self._pull_tensor.handle]
                avatar = push_avatar + pull_avatar
            else:
                real = [t.handle for t in self.parent._tensor] if isinstance(self.parent._tensor, (tuple, list)) else [self.parent._tensor.handle]
                avatar = [t.handle for t in self._tensor] if isinstance(self._tensor, (tuple, list)) else [self._tensor.handle]

        self._post_start_barrier(avatar, real)
        self._post_start_op(avatar)
        self._post_push_pull_barrier(avatar)

        # post real op
        if self.parent is None:
            self._post_communication(self._avatar)
        else:
            self._post_communication(self._tensor)

        self._post_end_op(avatar)

        self._post_end_barrier(avatar, real)

    # the push barrier is for barrier push-pull of all worker
    def _post_push_pull_barrier(self, avatar):
        if self.op == "push_pull":
            # push barrier and write dependency on barrier tensor and avatar with highest priority
            self._comm.push(self._barrier_key, self._barrier_tensor, -self.priority)
            deps = [self._barrier_tensor.handle] + avatar
            barrier_tensors_out = (NDArrayHandle * len(deps))(*deps)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_out, 0,
                barrier_tensors_out, len(deps),
                10000000 - self.priority))

    def _post_start_barrier(self, avatar, real):
        """The start barrier is for keeping the original dependency. It does not need any callback."""
        if self.parent is None:
            barrier_tensors_in = (NDArrayHandle * len(real))(*real)
            if self.op == "push_pull":
                tensor_out = [self._barrier_tensor.handle]
            else:
                tensor_out = avatar
            barrier_tensors_out = (NDArrayHandle * len(tensor_out))(*tensor_out)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_in, len(real),
                barrier_tensors_out, len(tensor_out),
                10000000-self.priority))
        else:
            if hasattr(self.parent, "_posted_start_barrier"):
                return
            self.parent._posted_start_barrier = True
            if self.op == "push_pull":
                push_pull_barriers = []
                for child in self.parent.children:
                    push_pull_barriers.append(child._barrier_tensor.handle)
                deps = real + push_pull_barriers
            else:
                children_tensors = []
                for child in self.parent.children:
                    if isinstance(child._tensor, (tuple, list)):
                        for t in child._tensor:
                            # including push tensor and pull tensor
                            if isinstance(t, (tuple, list)):
                                children_tensors += [tt.handle for tt in t]
                            else:
                                children_tensors += [t.handle]
                    else:
                        children_tensors += [child._tensor.handle]
                deps = real + children_tensors
            barrier_tensors_out = (NDArrayHandle * len(deps))(*deps)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_out, 0,
                barrier_tensors_out, len(deps),
                10000000-self.priority))

    def _post_start_op(self, avatar):
        """The start op is only for notifying the Core about task ready. It does not add any dependency to the
        original DAG."""
        if self._immediate:
            return

        def start_callback(on_complete):
            if self._immediate:
                # call on_complete directly
                check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                    c_void_p(on_complete)))
                return
            self._on_complete = on_complete
            self.notify_ready()

        # avoid garbage collection
        self._mxnet_start_callback = callback_t(start_callback)

        # post start op
        if self.op == "push_pull":
            tensor_out = (NDArrayHandle * 1)(*[self._barrier_tensor.handle])
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
                tensor_out, 0, tensor_out, 1, self._mxnet_start_callback, 1000000-self.priority))
        else:
            tensor_out = (NDArrayHandle * len(avatar))(*avatar)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
                tensor_out, 0, tensor_out, len(avatar), self._mxnet_start_callback, 1000000-self.priority))

    def _post_end_op(self, avatar):
        """The end op is only for notifying the Core about task finishing. It does not add any dependency to the
        original DAG."""
        def end_callback(on_complete):
            # call on_complete directly
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                c_void_p(on_complete)))
            self.notify_finish()

        # avoid garbage collection
        self._mxnet_end_callback = callback_t(end_callback)

        # post end op
        tensor_out = (NDArrayHandle * len(avatar))(*avatar)
        check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
            tensor_out, 0, tensor_out, len(avatar), self._mxnet_end_callback, 1000000-self.priority))

    def _post_end_barrier(self, avatar, real):
        """The end barrier is for keeping the original dependency. It does not need any callback."""
        if self.parent is None:
            deps = real + avatar
            barrier_tensors_out = (NDArrayHandle * len(deps))(*deps)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_out, 0,
                barrier_tensors_out, len(deps),
                10000000-self.priority))
        else:
            # _child_tensors is a list of avatar, and avatar itself is also a list
            if not hasattr(self.parent, "_children_tensors"):
                self.parent._children_tensors = [avatar]
            else:
                self.parent._children_tensors.append(avatar)
            if len(self.parent._children_tensors) == len(self.parent.children):
                tensors_in = [_ for sublist in self.parent._children_tensors for _ in sublist]
                barrier_tensors_in = (NDArrayHandle * len(tensors_in))(*tensors_in)
                barrier_tensors_out = (NDArrayHandle * len(real))(*real)
                check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                    barrier_tensors_in, len(tensors_in),
                    barrier_tensors_out, len(real),
                    10000000-self.priority))

    def _tensor_size(self):
        """Returns the size of one tensor of the task"""
        if self.op == "push_pull":
            assert isinstance(self._push_tensor, (tuple, list))
            return self._push_tensor[0].size
        else:
            if isinstance(self._tensor, (tuple, list)):
                return self._tensor[0].size
            else:
                return self._tensor.size

    def _partition_single_tensor(self, tensor, size):
        """Only partition a single tensor.

        Arguments:
            size: An integer. After partitioning, each tensor partition size must be equal or smaller than `size`.

        Returns:
            A list of partitioned tensors.
        """
        number = (tensor.size - 1) // size + 1
        if number > tensor.shape[0]:
            self._logger.warning(
                "The number of tensor rows (with shape {}) is smaller than partition number {}.".format(tensor.shape, number))
            number = tensor.shape[0]
        num_per_partition = tensor.shape[0] // number
        partitions_with_extra = tensor.shape[0] % number

        partitions = []
        start = 0
        end = num_per_partition
        for i in range(number):
            handle = NDArrayHandle()
            check_call(BYTESCHEDULER_LIB.bytescheduler_get_ndarray_avatar(
                tensor.handle, byref(handle)))
            avatar = NDArray(handle)[start:end]
            partitions.append(avatar)
            start = end
            end += num_per_partition
            if i >= number - partitions_with_extra - 1:
                end += 1
        return partitions

    def _partition_tensor_list(self, tensors, size):
        """Partition a list of tensors.

        Arguments:
            size: An integer. After partitioning, each tensor partition size must be equal or smaller than `size`.

        Returns:
            A list of partitioned tensors.
        """
        tot_partitions = []
        num_partitions = 0
        for tensor in tensors:
            partitions = self._partition_single_tensor(tensor, size)
            if num_partitions:
                assert num_partitions == len(partitions)
            else:
                num_partitions = len(partitions)
            tot_partitions.append(partitions)

        # Group partitions with same index from each tensor
        ret_partitions = []
        for p in zip(*tot_partitions):
            ret_partitions.append(p)
        return ret_partitions

    def _partition_tensor(self, size):
        """Zero-copy implementation.
        Note: ndarray works for up to ~4 billion parameters.

        Arguments:
            size: An integer. After partitioning, each tensor partition size must be equal or smaller than `size`.

        Returns:
            A list of partitioned tensors.
        """
        if self.op == "push_pull":
            assert isinstance(self._push_tensor, (tuple, list)) and isinstance(self._pull_tensor, (tuple, list))
            push_partitions = self._partition_tensor_list(self._push_tensor, size)
            pull_partitions = self._partition_tensor_list(self._pull_tensor, size)
            assert len(push_partitions) == len(pull_partitions)
            ret_partitions = []
            for p in zip(push_partitions, pull_partitions):
                ret_partitions.append(p)
            return ret_partitions
        else:
            if isinstance(self._tensor, (tuple, list)):
                return self._partition_tensor_list(self._tensor, size)
            else:
                return self._partition_single_tensor(self._tensor, size)

    def _immediate_do(self):
        self._prepare()

