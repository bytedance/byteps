#!/usr/bin/python

from __future__ import absolute_import
import time
import sys
import logging
from bytescheduler.common.bytecore import core
from bytescheduler.test.test_bytetask import TestTask


def test_core():
    core.start(rank=0, arch="ps")
    start = time.time()
    for i in reversed(range(3)):
        print("Post task {}".format(i))
        t = TestTask(i, 1000*(i+1), "allreduce", start_time=time.time(), add_notify_finish_trigger=True)
        t.priority = i
        core.post(t)
    core.shutdown(wait_for_all=True)
    print("Finished all tasks in {} seconds!".format(time.time() - start))


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("Usage: test_bytecore.py [debug_flag]")
        exit(1)

    if len(sys.argv) >= 2:
        logger = logging.getLogger("ByteScheduler")
        logger.setLevel(logging.DEBUG)

    test_core()
