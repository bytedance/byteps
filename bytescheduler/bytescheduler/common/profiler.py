from __future__ import absolute_import
import json
import time
import collections
import threading
try:
    import queue
except ImportError:
    import Queue as queue
import json


class Profiler(object):
    """
    The chrome trace format:
    https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit#heading=h.xqopa5m0e28f
    After profiling, open the trace using Chrome Browser.
    """
    def __init__(self, path):
        """
        Arguments:
            path: The path to save the trace file.
        """
        self._path = path
        self._enable = True if path else False
        self._begin = time.time()
        self._tensors = collections.OrderedDict()

        # Events are pushed to queue before writing to file by another thread.
        self._events = queue.Queue()
        self._cache = dict()

        # Start a writer thread
        self._writer = threading.Thread(target=self._export, args=())
        self._writer.start()

    def put(self, tensor, activity, ph):
        """Add one trace event.

        Arguments:
            tensor: The name of the tensor.
            activity: The event type, e.g., QUEUE, COMMUNICATION.
            ph: 'B' or 'E' indicating the start or the end of the event.
        """
        if not self._enable:
            return

        activity = activity.upper()
        if tensor not in self._tensors:
            self._tensors[tensor] = len(self._tensors)
            self._events.put(dict(
                name="process_name",
                ph="M",
                pid=self._tensors[tensor],
                args=dict(
                    name=tensor,
                )
            ))
            self._events.put(dict(
                name="process_sort_index",
                ph="M",
                pid=self._tensors[tensor],
                args=dict(
                    sort_index=self._tensors[tensor],
                )
            ))
            self._cache[tensor+activity] = None

        ts = int((time.time() - self._begin) * 10**6)

        if ph == "E" or ph == "e":
            self._events.put(self._cache[tensor+activity])
            self._cache[tensor+activity] = None
            self._events.put(dict(
                name=activity,
                ph=ph,
                ts=ts,
                pid=self._tensors[tensor],
            ))
        else:
            if tensor + activity in self._cache:
                assert self._cache[tensor+activity] is None
            self._cache[tensor+activity] = dict(
                name=activity,
                ph=ph,
                ts=ts,
                pid=self._tensors[tensor],
            )

    def stop(self):
        """Shutdown writer thread before exit."""
        self._enable = False
        self._writer.join()

    def _export(self):
        """Consume events in queue and dump events to json file."""
        if not self._enable:
            return

        with open(self._path, "w") as f:
            f.write('[')
        buf = ''
        while self._enable:
            event = self._events.get()
            buf = buf + json.dumps(event) + ', '

            # batch writing
            if len(buf) > 1000:
                with open(self._path, "a") as f:
                    f.write(buf)
                buf = ''

        # write remaining events
        with open(self._path, "a") as f:
            if len(buf) > 0:
                f.write(buf)
            f.write(']')
