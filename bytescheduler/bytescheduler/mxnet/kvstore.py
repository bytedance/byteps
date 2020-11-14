from __future__ import absolute_import
import mxnet as mx
import logging
from .kvstore_task import KVStoreTask
from ..common.bytecore import core


class ScheduledKVStore(mx.kvstore.KVStore):
    """An optimizer that wraps a mx.kvstore.KVStore, intercepting all init/push/pull operations and wrap as tasks.

    Usage example:
    ```
    from bytescheduler.mxnet.kvstore import ScheduledKVStore
    kv = mx.kvstore.create(args.kv_store)
    kv = ScheduledKVStore(kv)
    ```
    """

    def __init__(self, kvstore):
        """Construct a new kvstore, which uses MXNet kvstore under the hood for real communication.

        Args:
            kvstore: MXNet kvstore object.
        """
        self._logger = logging.getLogger("ByteScheduler")
        self._kvstore = kvstore
        self._rank = self._kvstore.rank

        # each worker schedules its tasks individually.
        self._immediate = False
        if self._rank != 0:
            self._immediate = True

        # Buffer a push request of each key
        self._push_buffer = dict()

        # Mapping string key to integer
        self._str_key_int = dict()

        # Count training step
        self._first_key = None
        self._step = 0

        # Start core
        core.start(rank=self._rank, arch="ps")
        
    def __del__(self):
        core.shutdown(True)
        self._kvstore.__del__()

    def __getattr__(self, item):
        return getattr(self._kvstore, item)

    def init(self, key, value):
        """Override the default one"""
        if not self._first_key:
            self._first_key = key
        self._str_key_int[key] = len(self._str_key_int)

        self._logger.debug("init {}".format(key))
        task = KVStoreTask(
            key,
            value,
            "init",
            priority=0,
            comm=self._kvstore,
            immediate=True,
            step=self._step,
            rank=self._rank,
            key_index=self._str_key_int[key],
        )
        core.post(task)

    def push(self, key, value, priority=0):
        """Override the default one.

        Key or value can be single element or list. In MXNet, key is a string and value is a list of NDArray with
        length equal to # of executors/devices
        """
        self._push_buffer[key] = (key, value, priority)

    def pull(self, key, out=None, priority=0, ignore_sparse=True):
        """Post each push/pull operation as a ByteTask to Core.

        Index is used as priority. We assume the closer to input layer, the smaller the index is. So smaller index
        indicates higher priority.
        """
        # Must be first pull for parameter initialization
        if key not in self._push_buffer:
            self._logger.debug("first pull of {}".format(key))
            task = KVStoreTask(
                key,
                out,
                "pull",
                priority=-priority,
                comm=self._kvstore,
                immediate=True,
                step=self._step,
                rank=self._rank,
                logger=self._logger,
                ignore_sparse=ignore_sparse,
            )
            core.post(task)
        else:
            if key == self._first_key:
                self._step += 1

            # Merge push and pull into one task
            (_key, _value, _priority) = self._push_buffer[key]
            task = KVStoreTask(
                key,
                (_value, out),
                "push_pull",
                priority=-priority,
                comm=self._kvstore,
                immediate=self._immediate,
                step=self._step,
                rank=self._rank,
                ignore_sparse=ignore_sparse,
                key_index=self._str_key_int[key],
            )
            del self._push_buffer[key]
            core.post(task)
