# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import ctypes
import os
import sysconfig
import atexit


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def get_extension_full_path(pkg_path, *args):
    assert len(args) >= 1
    dir_path = os.path.join(os.path.dirname(pkg_path), *args[:-1])
    full_path = os.path.join(dir_path, args[-1] + get_ext_suffix())
    return full_path


def check_extension(ext_name, ext_env_var, pkg_path, *args):
    full_path = get_extension_full_path(pkg_path, *args)
    if not os.path.exists(full_path):
        raise ImportError(
            'Extension %s has not been built.  If this is not expected, reinstall '
            'BytePS with %s=1 to debug the build error.' % (ext_name, ext_env_var))


class BytePSBasics(object):
    """Wrapper class for the basic BytePS API."""

    def __init__(self, pkg_path, *args):
        full_path = get_extension_full_path(pkg_path, *args)
        self.C_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

    def init(self, lazy=True):
        """A function that inits BytePS."""
        atexit.register(self.shutdown)
        if lazy:
            return self.C_LIB_CTYPES.byteps_lazy_init()
        else:
            return self.C_LIB_CTYPES.byteps_init()

    def shutdown(self):
        """A function that shuts BytePS down."""
        return self.C_LIB_CTYPES.byteps_shutdown()

    def suspend(self):
        """A function that suspends BytePS for elastic training."""
        return self.C_LIB_CTYPES.byteps_suspend()

    def resume(self, num_workers, num_servers, global_rank, context=None):
        """A function that restarts BytePS after being suspended, for elastic training."""
        # set DMLC environment variables here
        os.environ['DMLC_NUM_WORKER'] = str(num_workers)
        os.environ['DMLC_NUM_SERVER'] = str(num_servers)
        os.environ['BYTEPS_GLOBAL_RANK'] = str(global_rank)
        return self.C_LIB_CTYPES.byteps_resume(num_workers, num_servers)

    def size(self):
        """A function that returns the number of BytePS processes.
        Returns:
          An integer scalar containing the number of BytePS processes.
        """
        size = self.C_LIB_CTYPES.byteps_size()
        if size == -1:
            raise ValueError(
                'BytePS has not been initialized; use bps.init().')
        return size

    def local_size(self):
        """A function that returns the number of BytePS processes within the
        node the current process is running on.
        Returns:
          An integer scalar containing the number of local BytePS processes.
        """
        local_size = self.C_LIB_CTYPES.byteps_local_size()
        if local_size == -1:
            raise ValueError(
                'BytePS has not been initialized; use bps.init().')
        return local_size

    def rank(self):
        """A function that returns the BytePS rank of the calling process.
        Returns:
          An integer scalar with the BytePS rank of the calling process.
        """
        rank = self.C_LIB_CTYPES.byteps_rank()
        if rank == -1:
            raise ValueError(
                'BytePS has not been initialized; use bps.init().')
        return rank

    def local_rank(self):
        """A function that returns the local BytePS rank of the calling process, within the
        node that it is running on. For example, if there are seven processes running
        on a node, their local ranks will be zero through six, inclusive.
        Returns:
          An integer scalar with the local BytePS rank of the calling process.
        """
        local_rank = self.C_LIB_CTYPES.byteps_local_rank()
        if local_rank == -1:
            raise ValueError(
                'BytePS has not been initialized; use bps.init().')
        return local_rank

    def get_pushpull_speed(self):
        """A function that returns the current push pull speed. Speed is
        calculated every 10 seconds.
          Returns:
            A tuple: (ms since epoch, speed in MegaBytes per second)
        """
        pushpull_speed = self.C_LIB_CTYPES.byteps_get_pushpull_speed
        pushpull_speed.restype = ctypes.py_object
        entry = pushpull_speed()
        return entry
