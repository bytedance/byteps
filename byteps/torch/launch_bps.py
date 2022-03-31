#!/usr/bin/python

from __future__ import print_function
import os
import subprocess
import threading
import sys
import byteps.torch as bps


class PropagatingThread(threading.Thread):
    """ propagate exceptions to the parent's thread
    refer to https://stackoverflow.com/a/31614591/9601110
    """

    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                #  python 2.x
                self.ret = self._Thread__target(
                    *self._Thread__args, **self._Thread__kwargs)
            else:
                # python 3.x
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.exc


def launch_server(local_rank):
    if local_rank != 0:
        return
    def server_runner():
        my_env = os.environ.copy()
        my_env['DMLC_ROLE'] = 'server'
        my_env['DMLC_WORKER_ID'] = os.environ['DMLC_WORKER_ID']

        command = "echo starting bps server;python3 -c 'import byteps.server'; echo stop bps server"
        subprocess.check_call(command, env=my_env,
                            stdout=sys.stdout, stderr=sys.stderr, shell=True)
    t = PropagatingThread(target=server_runner)
    t.setDaemon(True)
    t.start()


def launch_scheduler(local_rank):
    if os.environ['DMLC_WORKER_ID'] != '0':
        return
    if local_rank != 0:
        return

    def scheduler_runner():
        my_env = os.environ.copy()
        my_env['DMLC_ROLE'] = 'scheduler'
        command = "echo starting bps scheduler;python3 -c 'import byteps.server';echo stop bps scheduler"
        subprocess.check_call(command, env=my_env,
                          stdout=sys.stdout, stderr=sys.stderr, shell=True)
    t = PropagatingThread(target=scheduler_runner)
    t.setDaemon(True)
    t.start()


def setup_bps_env(local_rank):
    gpus_list = os.environ["NVIDIA_VISIBLE_DEVICES"]
    gpu_per_node = len(gpus_list.split(","))

    os.environ['BYTEPS_LOCAL_RANK'] = str(local_rank)
    os.environ['BYTEPS_LOCAL_SIZE'] = str(gpu_per_node)


def launch_bps(local_rank):
    setup_bps_env(local_rank)
    launch_server(local_rank)
    launch_scheduler(local_rank)
    bps.init(local_rank)