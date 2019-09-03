from __future__ import absolute_import
import os
from ctypes import CDLL, RTLD_GLOBAL
from ..common import get_ext_suffix
from ..common.bytetask import ByteTask
import torch

# Load c_lib.so
dll_path = os.path.join(os.path.dirname(__file__),
                        'c_lib' + get_ext_suffix())
BYTESCHEDULER_LIB = CDLL(dll_path, RTLD_GLOBAL)


class HorovodTask(ByteTask):

    def _do(self):
        """Start allreduce."""
        handle, ctx = self._comm.allreduce_grad_async(self._tensor, self.name)
        self._comm.event_queue.put((self.kwargs['parameter'], self._tensor, handle, ctx, self._finish_callback))

    def _finish_callback(self):
        """Notify Core about completion of a tensor.

        Returns:
            a boolean value indicating whether the task or parent task is finished or not.
        """
        if self.parent is not None:
            parent_finish = self.notify_finish()
            return parent_finish
        else:
            self.notify_finish()
            return True

    def _prepare(self):
        """Add a CUDA query event to check the readiness of a tensor, i.e., when backward propagation is finished."""
        device = self._tensor.device
        # CPU device id is -1
        device_id = -1
        if device.index is None:
            assert device.type == "cpu"
        else:
            assert device.type == "cuda"
            device_id = device.index

        handle = BYTESCHEDULER_LIB.bytescheduler_create_event(device_id)
        self._comm.event_queue.put((self.kwargs['parameter'], self._tensor, handle, "READYEVENT", self.notify_ready))

    def _tensor_size(self):
        """Returns the number of parameters of the tensor."""
        size = 1
        for s in list(self._tensor.size()):
            size = size * s
        return size

    def _partition_tensor_v2(self, partition_list):
        """Partition a tensor according to a partition size list.

        Arguments:
            size: a list of integers indicating the relative size of each partition.

        Returns:
            A list of partitioned tensors.
        """
        assert partition_list is not None, "partition list can not be None."
        partition_list_merge = []
        for key in partition_list:
            if key != -1:
                partition_list_merge.append(key)
        partition_list = partition_list_merge

        self._logger.debug(
            "call v2 partition func for key {}, partition list: {}".format(self.name, partition_list))
        number = sum(partition_list)
        if number > self._tensor.shape[0]:
            self._logger.warning(
                "The number of tensor rows (with shape {}) is smaller than partition number {}.".format(self._tensor.shape, number))
            number = self._tensor.shape[0]

        num_per_partition = self._tensor.shape[0] // number
        partitions = []
        start = 0
        for i in range(len(partition_list)):
            end = num_per_partition * partition_list[i] + start
            if i == len(partition_list) - 1:
                avatar = self._tensor[start:]
            else:
                avatar = self._tensor[start:end]
            partitions.append(avatar)
            start = end
        return partitions

    def _partition_tensor(self, size):
        """Partition a tensor evenly.

        Arguments:
            size: An integer. After partitioning, each tensor partition size must be equal or smaller than `size`.

        Returns:
            A list of partitioned tensors.
        """
        number = (self._tensor_size() - 1) // size + 1
        if number > self._tensor.shape[0]:
            self._logger.warning(
                "The number of tensor rows (with shape {}) is smaller than partition number {}.".format(self._tensor.shape, number))
            number = self._tensor.shape[0]

        num_per_partition = self._tensor.shape[0] // number
        partitions_with_extra = self._tensor.shape[0] % number
        partitions = []
        start = 0
        end = num_per_partition
        for i in range(number):
            avatar = self._tensor[start:end]
            partitions.append(avatar)
            start = end
            end += num_per_partition
            if i >= number - partitions_with_extra - 1:
                end += 1
        return partitions

    def _immediate_do(self):
        self._do()


