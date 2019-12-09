#!/usr/bin/python

from __future__ import absolute_import
import sys

try:
    import queue
except ImportError:
    import Queue as queue
import threading
import logging
import collections
from .bytetask import ByteTask
from .tuner import Tuner
from .profiler import Profiler
import os


class ByteCore(object):
    """The core of ByteScheduler. Once Core gets a ByteTask (which represents a communication operation, e.g., push,
    allreduce), it partitions the ByteTask and decides when to send each partition according to priority."""

    def __init__(self, logger=None):
        """
        Args:
            logger: ByteScheduler logger object
        """
        if logger is None:
            self._logger = logging.getLogger("ByteScheduler")
        else:
            self._logger = logger

        # A priority queue of ByteTask, tasks are sorted according to its priority.
        self._queue = queue.PriorityQueue()

        # Scheduler thread
        self._scheduler = threading.Thread(target=self._loop, args=())
        self._scheduler.daemon = True
        self._is_started = False

        # DATA represents normal tasks and EXIT signals the scheduler thread to be terminated.
        self._commands = {'DATA': 0, 'EXIT': 1}

        # Control credit
        self._condition = threading.Condition(threading.Lock())

        # Pending tasks that are not ready
        self._pending = set()
        self._pending_lock = threading.Lock()

        # Only used to avoid task being garbage collected before completion.
        self._running = set()
        self._finished = collections.OrderedDict()

        # The rank of a worker
        self._rank = None

        # The communication architecture used, e.g., ps or allreduce.
        self._arch = None

        # Partition unit, i.e., the number of parameters
        self._partition = int(os.environ.get('BYTESCHEDULER_PARTITION', 1000000))

        # Credit, i.e., the max number of unacknowledged parameters
        self._credit = float(os.environ.get('BYTESCHEDULER_CREDIT', 4000000))
        self._credit_limit = self._credit

        # We expect that the first key is same across iterations and we use it to count how many training steps have
        # been run.
        self._first_key = None
        self._step = 0

        # Tuning
        self._credit_tuning = int(os.environ.get('BYTESCHEDULER_CREDIT_TUNING', 1))
        self._partition_tuning = int(os.environ.get('BYTESCHEDULER_PARTITION_TUNING', 0))
        self._tuner = None

        # hyper parameters of auto-tuning.
        self._current_point = {
            "credit": self._credit,
        }
        self._next_point = None

        # profiling
        self._timeline = os.environ.get('BYTESCHEDULER_TIMELINE', '')
        self._profiler = None

    def start(self, rank, arch):
        """Start core.
        Args:
            rank: the rank of the worker
            arch: the communication architecture, "ps" or "allreduce"
        """
        if self._is_started:
            self._logger.warning("Core is already started.")
            return

        self._rank = rank

        # Setup profiler
        if self._rank == 0 and self._timeline:
            self._logger.info("rank {}: profiler is enabled.".format(self._rank))
            self._profiler = Profiler(self._timeline)
        else:
            self._profiler = Profiler('')

        assert arch == "ps" or arch == "allreduce", arch + " not supported!"
        self._arch = arch

        # Support tuning partition for allreduce
        if self._partition_tuning:
            assert arch == "allreduce", "Do not support partition tuning for ps."
            self._current_point["partition"] = self._partition

        if (rank == 0 and self._credit_tuning) or self._partition_tuning:
            self._tuner = Tuner(rank=self._rank, arch=arch, credit_tuning=self._credit_tuning,
                                partition_tuning=self._partition_tuning, logger=self._logger)

        self._scheduler.start()
        self._is_started = True

        self._logger.info(
            "start Core {}: credit {}, partition {}, credit tuning {}, partition tuning {}.".format(
                self._rank, self._credit, self._partition, self._credit_tuning, self._partition_tuning))

    def shutdown(self, wait_for_all=False):
        """Shut Core down.

        Args:
            wait_for_all: Flag indicating whether to wait completion of undone tasks.
        """
        if not self._is_started:
            self._logger.warning("Core is already shutdown.")
            return
        if wait_for_all:
            self._queue.put((sys.maxint, self._commands['EXIT'], None))
        else:
            self._queue.put((-sys.maxint, self._commands['EXIT'], None))
        with self._condition:
            self._credit = sys.maxint
            self._condition.notify_all()
        self._scheduler.join()
        self._is_started = False
        self._tuner.exit()
        self._profiler.stop()
        self._logger.info("shutdown Core {}.".format(self._rank))

    def post(self, task):
        """Post a communication task to Core for scheduling.
        Args:
            task: a ByteTask object
        Returns:
            A boolean value indicating whether the task is successfully posted
        """
        if not self._is_started:
            self._logger.error("Core is not running, call start first!")
            return False

        if not isinstance(task, ByteTask):
            self._logger.error(
                "{} is not an instance of ByteTask!".format(task.desc))
            return False
        else:
            # Set the first key and use it to count number of training steps.
            if not self._first_key:
                self._first_key = task.name
            if self._first_key == task.name:
                self._step += 1
                if self._tuner:
                    self._tune()

            # Partition a task if its tensor is larger than a threshold.
            if task.tensor_size() > self._partition:
                subtasks = task.partition(size=self._partition)
            else:
                subtasks = [task]

            # A task will bypass scheduling and start immediately after partition if immediate is True.
            if task.is_immediate():
                # The callback runs after an immediate task is finished.
                def _end_callback(t, self):
                    with self._condition:
                        self._running.remove(t)
                        self._finished[t.name] = t
                    self._profiler.put(t.name, t.op + 'COMMUNICATION', 'E')

                for t in subtasks:
                    with self._condition:
                        self._running.add(t)
                    t.immediate_do(callback=_end_callback, callback_context=self)
                    self._profiler.put(t.name, t.op + 'COMMUNICATION', 'B')
                return True

            # The callback runs when a non-immediate task is ready.
            def _start_callback(task, self):
                with self._pending_lock:
                    self._pending.remove(task)
                self._profiler.put(task.name, task.op + 'QUEUE', 'B')
                with self._condition:
                    self._queue.put((task.priority, self._commands['DATA'], task))
                    self._condition.notify_all()
                self._logger.debug(
                    "{} has been posted into Core with priority {}".format(task.desc, task.priority))

            # Prepare the task, i.e., add dependency Proxies.
            for t in subtasks:
                with self._pending_lock:
                    self._pending.add(t)
                t.prepare(callback=_start_callback, callback_context=self)
            return True

    def _loop(self):
        """The main scheduling logic is a while loop that pops a task from queue each time and do it if credit is enough.
        The credit decreases when a task is running and increases when a task is finished.
        """

        # The callback runs when a non-immediate task is finished.
        def _end_callback(task, self):
            with self._condition:
                self._credit += task.tensor_size()
                if self._credit > self._credit_limit:
                    self._credit = self._credit_limit
                self._running.remove(task)
                self._condition.notify_all()
            self._finished[task.name] = task
            self._profiler.put(task.name, task.op + 'COMMUNICATION', 'E')

        while True:
            with self._condition:
                while True:
                    try:
                        priority, cmd, task = self._queue.get(False)
                    except:
                        # wait for (potential) new task
                        self._condition.wait()
                        continue
                    if task and self._credit <= 0:
                        self._queue.put((priority, cmd, task))
                        # wait for (potential) new credit
                        self._condition.wait()
                    else:
                        break

            if cmd == self._commands['EXIT']:
                break
            else:
                self._profiler.put(task.name, task.op + 'QUEUE', 'E')
                with self._condition:
                    self._running.add(task)
                    self._credit -= task.tensor_size()
                task.do(callback=_end_callback, callback_context=self)
                self._profiler.put(task.name, task.op + 'COMMUNICATION', 'B')

    def _tune(self):
        if self._tuner.stopped and self._next_point is None:
            self._tuner.exit()
            return
        # Only rank 0 runs auto-tuning algorithm
        if self._rank == 0:
            self._tuner.record(self._current_point, self._step)
        if self._next_point is None:
            self._next_point = self._tuner.next_point()
        if self._next_point is not None and self._step == self._next_point["step"]:
            with self._condition:
                if "credit" in self._next_point:
                    self._credit_limit = self._next_point["credit"]
                    self._credit = self._next_point["credit"]
                    self._logger.info("core {}: autotuning sets credit to {}K at training step {}.".format(
                            self._rank, int(self._credit / 1000), self._step))
                if "partition" in self._next_point:
                    self._partition_unit = self._next_point["partition"]
                    self._logger.info("core {}: autotuning sets partition to {}K at training step {}.".format(
                            self._rank, int(self._partition / 1000), self._step))
                self._current_point = self._next_point
                self._next_point = None

# Init a core once the module is imported
core = ByteCore()
