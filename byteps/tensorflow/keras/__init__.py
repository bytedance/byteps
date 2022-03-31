# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
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
# ==============================================================================

import inspect

import tensorflow as tf

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) >= LooseVersion("1.4.0"):
    from tensorflow import keras
    from tensorflow.python.keras import backend as K
else:
    from tensorflow.contrib import keras
    from tensorflow.contrib.keras import backend as K

from byteps.tensorflow import init
from byteps.tensorflow import shutdown
from byteps.tensorflow import size
from byteps.tensorflow import local_size
from byteps.tensorflow import rank
from byteps.tensorflow import local_rank
from byteps.tensorflow import Compression

import byteps._keras as _impl
from byteps.tensorflow.keras import callbacks


def DistributedOptimizer(optimizer, name=None,
                         device_dense='', device_sparse='',
                         compression=Compression.none,
                         sparse_as_dense=False):
    """
    An optimizer that wraps another keras.optimizers.Optimizer, using an push_pull to
    average gradient values before applying gradients to model weights.
    Args:
        optimizer: Optimizer to use for computing gradients and applying updates.
        name: Optional name prefix for the operations created when applying
              gradients. Defaults to "Distributed" followed by the provided
              optimizer type.
        device_dense: Device to be used for dense tensors. Uses GPU by default.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
        sparse_as_dense: Treat all sparse gradients as dense tensors.  This can
                         help improve performance and memory utilization if
                         the original sparse gradient has high density.
                         Defaults to false.
    """
    return _impl.create_distributed_optimizer(keras, optimizer, name,
                                              device_dense, device_sparse, compression,
                                              sparse_as_dense)


def broadcast_global_variables(root_rank):
    """Broadcasts all global variables from root rank to all other processes.
    Arguments:
        root_rank: Rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    return _impl.broadcast_global_variables(K, root_rank)


def push_pull(value, name=None, average=True):
    """
    Perform an push_pull on a tensor-compatible value.
    Arguments:
        value: A tensor-compatible value to reduce.
               The shape of the input must be identical across all ranks.
        name: Optional name for the constants created by this operation.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.
    """
    return _impl.push_pull(K, value, name, average)


def broadcast(value, root_rank, name=None):
    """
    Perform a broadcast on a tensor-compatible value.
    Arguments:
        value: A tensor-compatible value to reduce.
               The shape of the input must be identical across all ranks.
        root_rank: Rank of the process from which global variables will be
                   broadcasted to all other processes.
        name: Optional name for the constants created by this operation.
    """
    return _impl.broadcast(K, value, root_rank, name)


def load_model(filepath, custom_optimizers=None, custom_objects=None, compression=Compression.none):
    """
    Loads a saved Keras model with a BytePS DistributedOptimizer.
    The DistributedOptimizer will wrap the underlying optimizer used to train
    the saved model, so that the optimizer state (params and weights) will
    be picked up for retraining.
    By default, all optimizers in the module `keras.optimizers` will be loaded
    and wrapped without needing to specify any `custom_optimizers` or
    `custom_objects`.
    # Arguments
        filepath: One of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        custom_optimizers: Optional list of Optimizer subclasses to support
            during loading.
        custom_objects: Optional dictionary mapping names (strings) to custom
            classes or functions to be considered during deserialization.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
    # Returns
        A Keras model instance.
    # Raises
        ImportError: If h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    def wrap_optimizer(cls):
        return lambda **kwargs: DistributedOptimizer(cls(**kwargs), compression=compression)
    optimizer_modules = {keras.optimizers.Optimizer.__module__}
    return _impl.load_model(keras, wrap_optimizer, optimizer_modules, filepath, custom_optimizers, custom_objects)
