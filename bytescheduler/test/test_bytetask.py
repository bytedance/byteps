#!/usr/bin/python

from __future__ import absolute_import
import time
import sys
import logging
from bytescheduler.common.bytetask import ByteTask


class TestTask(ByteTask):

    def _additional_init(self):
        self.start_time = self.kwargs["start_time"]
        return

    def _prepare(self):
        self.notify_ready()
        return

    def _do(self):
        return

    def _wait_until_finish(self):
        need_time = self.tensor_size() / 1000.0
        time.sleep(max(0, self.start_time + need_time - time.time()))
        return

    def _tensor_size(self):
        return self._tensor

    def _partition_tensor(self, size):
        partitions = []
        number = self.tensor_size() / size
        for i in range(number):
            partitions.append(self.tensor_size() * 1.0 / number)
        return partitions

    def _notify_upper_layer_finish(self):
        print("Finished task {} in {} seconds!".format(self.name, time.time() - self.start_time))
        return


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: test_bytetask.py tensor_size partition_size [debug_flag]")
        exit()

    if len(sys.argv) >= 4:
        logger = logging.getLogger("ByteScheduler")
        logger.setLevel(logging.DEBUG)

    t = TestTask("test", int(sys.argv[1]), "allreduce", 
                 start_time=time.time(), add_notify_finish_trigger=True)
    t.partition(size=int(sys.argv[2]))

    t.do()
