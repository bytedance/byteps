#!/usr/bin/python

from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from threading import Thread, Lock, Condition
import logging


class ByteTask(with_metaclass(ABCMeta)):
    """ A unified communication operation (e.g., push, pull, allreduce) abstraction for ALL ML frameworks and
    communication methods."""

    def __init__(self, name, tensor, op, 
                 priority=0, comm=None, parent=None, max_partition=1000,
                 add_notify_finish_trigger=False, immediate=False, step=0, partition_index=-1, rank=0,
                 logger=None, **kwargs):
        """ByteTask initialization.

        Args:
            name: The name of task, typically using the key index.
            tensor: The tensors of the communication task.
            op: The type of the communication, e.g., allreduce, push, pull.
            priority: The priority of the task. Small number indicates higher priority.
            comm: The communication object, e.g., kvstore.
            parent: The parent ByteTask object, if have.
            max_partition: The max number of children tasks.
            add_notify_finish_trigger: True if the ML engine runs the task in a blocking way.
            immediate: True if it is an immediate task, i.e., no priority scheduling is required.
            step: Which training step the task belongs to.
            partition_index: If the task is a child task, it has a unique index.
            rank: The rank of worker.
            logger: The logging object.
            kwargs: Other necessary arguments.
        """
        self.name = str(name)
        self.op = op
        if parent is None:
            self.priority = int(priority) * max_partition
        else:
            self.priority = int(priority)
        self.parent = parent
        self.kwargs = kwargs
        self.children = None
        self.partition_count = 1
        self.done_count = 0
        self._done_count_lock = Lock()

        # Can be a single tensor or a list of tensors
        self._tensor = tensor
        self._comm = comm

        # Task states: posted -> ready -> started -> finished
        self._ready = False
        self._started = False

        # The lock is for blocking wait_until_finish()
        # will only be released in notify_finish()
        self._finished = False
        self._lock_before_finish = Condition(Lock())

        # Wait_until_finish() thread
        self._thread = None

        # Callback for Core calling prepare()
        self._prepare_callback = None
        self._prepare_callback_context = None        

        # Callback for Core calling do()
        self._do_callback = None
        self._do_callback_context = None

        self._add_notify_finish_trigger = add_notify_finish_trigger
        self._immediate = immediate
        self._partition_index = partition_index
        self._step = step
        self._rank = rank

        # Logger
        if logger is None:
            self._logger = logging.getLogger("ByteScheduler")
        else:
            self._logger = logger

        self.desc = "rank {} step {} task {} {}".format(self._rank, self._step, self.op, self.name)
        self._additional_init()

        self._logger.debug(
            "{} was created".format(self.desc))
        return

    def prepare(self, callback=None, callback_context=None):
        """Prepare a task with dependencies, callback and callback_context are for Core.

        Args:
            callback: a function that runs once the task is ready.
            callback_context: context of callback function, typically Core.
        """
        self._prepare_callback = callback
        self._prepare_callback_context = callback_context
        if self.children is not None:
            for child in self.children:
                child.prepare(callback, callback_context)
        else:
            self._prepare()
        
        self._logger.debug(
            "{} prepare() started".format(self.desc))
        return

    def do(self, callback=None, callback_context=None):
        """Run a task, callback and callback_context are for Core.

        Core calls this interface to let engines and the underlying communication stacks send the tensor.

        Args:
            callback: a function that runs once the task is finished.
            callback_context: context of callback function, typically Core.

        """
        self._do_callback = callback
        self._do_callback_context = callback_context

        if self.children is not None:
            for child in self.children:
                if not child.is_started():
                    child.do(callback, callback_context)
        else:
            self._do()
            if self._add_notify_finish_trigger:
                self.add_notify_finish_trigger()

        self._started = True
        if self.parent is not None:
            self.parent._started = True

        self._logger.debug(
            "{} do() started".format(self.desc))
        return

    def add_notify_finish_trigger(self):
        """In the case that underlying engine only provides blocking wait but does not provide a callback."""
        def background(task):
            task._wait_until_finish()
            task.notify_finish()

        self._thread = Thread(target=background, args=[self])
        self._thread.start()

    def wait_until_finish(self):
        """Block until the task is finished"""
        if self.children is not None:
            for child in self.children:
                if not child.is_finished():
                    child.wait_until_finish()
        else:
            # We can get this lock only after notify_finish()
            with self._lock_before_finish:
                self._lock_before_finish.wait()
        return

    def immediate_do(self, callback=None, callback_context=None):
        """Called without priority scheduling"""
        self._do_callback = callback
        self._do_callback_context = callback_context
        return self._immediate_do()

    def is_immediate(self):
        """Check if the task is an immediate task, return True if the task does not need priority scheduling.
        Returns:
            A boolean value indicating whether it is an immediate task.
        """
        return self._immediate

    def is_ready(self):
        """Check if the task is ready for scheduling, i.e., the tensors of the task are ready for communication.
        Returns:
            A boolean value indicating whether the task is ready for scheduling.
        """
        return self._ready

    def is_started(self):
        """Check if a task is already started.
        Returns:
            A boolean value indicating whether the task is started.
        """
        return self._started

    def is_finished(self):
        """Check if a task is finished.
       Returns:
            A boolean value indicating whether the task is finished.
        """
        return self._finished

    def notify_ready(self):
        """Notify Core that the task is ready for scheduling.

        Most ML frameworks are asynchronous -- when a communication operation is posted to the engine, possibly the
        tensor has not been computed or ready to be sent. So the ML engine needs to notify Core about a tensor being
        ready, so that Core can actually begin scheduling it.
        """
        self._ready = True
        self._logger.debug(
            "{} is ready".format(self.desc))
        if self._prepare_callback is not None:
            self._prepare_callback(self, self._prepare_callback_context)
        return

    def notify_finish(self):
        """Notify Core that the task is already finished.

        Once the communication of a tensor (all-reduce, push or pull) has been finished, the framework engine must
        notify Core about this, so that Core can continue scheduling more tasks.
        """
        if self.parent is not None:
            parent_finish = self.parent.partition_done()
        else:
            parent_finish = True
            self._notify_upper_layer_finish()
        
        if self._do_callback is not None:
            self._do_callback(self, self._do_callback_context)

        self._finished = True
        with self._lock_before_finish:
            self._lock_before_finish.notify()

        self._logger.debug(
            "{} has finished".format(self.desc))
        return parent_finish

    def partition_done(self):
        """Called by parent when a task partition is finished.
        Returns:
            A boolean value indicating whether all task partitions are finished.
        """
        assert self.children is not None, "{} unexpected partition_done()".format(self.desc)
        self._done_count_lock.acquire()
        self.done_count += 1
        self._logger.debug(
            "{} done_count = {}".format(self.desc, self.done_count))
        finish = False
        if self.done_count == self.partition_count:
            finish = True
            self.notify_finish()

        self._done_count_lock.release()
        return finish

    def tensor_size(self):
        """Get the tensor size, i.e., the number of parameters of one tensor of the task
        Returns:
            An integer scalar with the size of a tensor of the task.
        """
        return self._tensor_size()

    def partition(self, size=None):
        """Partition a task into small onces

        Core calls this interface to partition a task and its tensor into one or multiple tasks with tensors no larger
        than size. All popular frameworks provide zero-copy APIs for partitioning a tensor.

        Args:
            size: an integer indicating the max size of each partition or a tuple/list of integers indicating the size
            of each partition.

        Returns:
            A list of partitioned children tasks
        """
        # A task can only be partitioned once
        assert self.children is None, "{} has been partitioned before".format(self.desc)

        # For now, we only support one layer of partitioning
        assert self.parent is None, "{} is already a partition".format(self.desc)

        # Check size not None
        assert size is not None and isinstance(size, (int, tuple, list)), \
            "{} got unclear partition request".format(self.desc)

        # Partition tasks evenly or based on a list
        if isinstance(size, list):
            tensor_partitions = self._partition_tensor_v2(size)
        else:
            tensor_partitions = self._partition_tensor(size)
        
        self.children = []
        for i in range(len(tensor_partitions)):
            self.children.append(
                self.__class__(
                    self.name + "_" + str(i),
                    tensor_partitions[i],
                    self.op,
                    priority=(self.priority + i),
                    comm=self._comm,
                    parent=self,
                    add_notify_finish_trigger=self._add_notify_finish_trigger,
                    immediate=self._immediate,
                    step=self._step,
                    partition_index=i,
                    rank=self._rank,
                    logger=self._logger,
                    **self.kwargs
                )
            )
        
        self.partition_count = len(tensor_partitions)
        self._logger.debug(
            "{} is partitioned into {}".format(self.desc, len(tensor_partitions)))
        return self.children

    @abstractmethod
    def _prepare(self):
        """Must be overriden -- required by Core.

        If a task is partitioned, only the children will call this function. The implementation depends on the underlying
        engine scheduler.

        No return value, non-blocking.
        """
        pass

    @abstractmethod
    def _do(self):
        """Must be overriden -- required by Core.

        If a task is partitioned, only the children will call this. This function depends on the underlying
        communication method.

        No return value, non-blocking.
        """
        pass


    @abstractmethod
    def _tensor_size(self):
        """Must be overriden -- required by Core.

        This function depends on the data structure of self._tensor.

        Must return an integer, i.e., the size of self._tensor.
        """
        pass

    @abstractmethod
    def _partition_tensor(self, size):
        """Must be overriden -- required by Core.

        If a task is partitioned, only the parent will call this. This function depends on the data structure of
        self._tensor.

        Must return a list of (partitioned) tensors.
        """
        pass

    def _partition_tensor_v2(self, size):
        """Optional -- required by allreduce.

        If a task is partitioned, only the parent will call this. This function depends on the data structure of
        self._tensor.

        Must return a list of (partitioned) tensors.
        """
        return

    def _additional_init(self):
        """Optional -- may be required by framework.

        If we have any framework-specific work to do during __init__. This function will be called at the end of __init__.

        No return value, blocking.
        """
        self._logger.debug(
            "{} _additional_init is not implemented".format(self.desc))
        return

    def _wait_until_finish(self):
        """Optional -- may be required by framework.

        If a task is partitioned, only the children will call this. This function depends on the underlying
        communication method.

        No return value, blocking.
        """
        self._logger.debug(
            "{} _wait_until_finish is not implemented".format(self.desc))
        return

    def _notify_upper_layer_finish(self):
        """Optional -- may be required by framework.

        If a task is partitioned, only the parent will call this. This function depends on the upper layer framework.

        No return value.
        """
        self._logger.debug(
            "{} _notify_upper_layer_finish is not implemented".format(self.desc))
        return

    def _immediate_do(self):
        """Optional -- may benefit certain framework.

        If a task is partitioned, only the children will call this. This function lets Core bypass all scheduling for
        this task.

        No return value, non-blocking.
        """
        pass
