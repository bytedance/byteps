# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

import byteps.tensorflow as bps
import tensorflow as tf

def create_distributed_optimizer(keras, optimizer, name, device_dense, device_sparse,
                                 compression, sparse_as_dense):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        _HAS_AGGREGATE_GRAD = True
        def __init__(self, **kwargs):
            self._name = name or "Distributed%s" % self.__class__.__base__.__name__
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense
            self._aggregated_gradients = False
            super(self.__class__, self).__init__(**kwargs)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.
            See Optimizer.get_gradients() for more info.
            In DistributedOptimizer, get_gradients() is overriden to also
            push_pull the gradients before returning them.
            """
            gradients = super(self.__class__, self).get_gradients(loss, params)
            return self._push_pull(gradients)

        def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
            """
            Compute gradients of all trainable variables.
            See Optimizer.get_gradients() for more info.
            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            tape = backprop.GradientTape() if tape is None else tape
            grads_and_vars = super(self.__class__, self)._compute_gradients(
                # pylint: disable=protected-access
                loss,
                var_list,
                grad_loss,
                tape=tape)
            grads, weights = list(zip(*grads_and_vars))
            grads = self._push_pull(grads)
            return list(zip(grads, weights))

        def _push_pull(self, gradients):
            self._aggregated_gradients = True
            if bps.size() > 1:
                averaged_gradients = []
                with tf.name_scope(self._name + "_Push_Pull") as scope:
                    for grad in gradients:
                        if grad is not None:
                            if isinstance(grad, tf.IndexedSlices):
                                if self.sparse_as_dense:
                                    return tf.convert_to_tensor(grad)
                                else:
                                    raise ValueError(
                                        "IndexedSlices are not supported when "
                                        "`backward_passes_per_step` > 1 and "
                                        "`sparse_as_dense` is False."
                                    )
                            avg_grad = bps.push_pull(grad, scope,
                                                     device_dense=self._device_dense,
                                                     device_sparse=self._device_sparse,
                                                     compression=self._compression)
                            averaged_gradients.append(avg_grad)
                        else:
                            averaged_gradients.append(None)
                    return averaged_gradients
            else:
                return gradients

        def apply_gradients(self, *args, **kwargs):
            if not self._aggregated_gradients:
                raise Exception('`apply_gradients()` was called without a call to '
                                '`get_gradients()` or `_aggregate_gradients`. If you\'re '
                                'using TensorFlow 2.0, please specify '
                                '`experimental_run_tf_function=False` in `compile()`.')
            return super(self.__class__, self).apply_gradients(*args, **kwargs)

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an push_pull implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without BytePS.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls.from_config(optimizer.get_config())


def _eval(backend, op_or_result):
    if bps._executing_eagerly():
        return op_or_result
    else:
        return backend.get_session().run(op_or_result)


if hasattr(bps, 'broadcast_global_variables'):
    def broadcast_global_variables(backend, root_rank):
        return _eval(backend, bps.broadcast_global_variables(root_rank))


def push_pull(backend, value, name, average):
    return _eval(backend,  bps.push_pull(tf.constant(value, name=name), average=average))


def broadcast(backend, value, root_rank, name):
    return _eval(backend, bps.broadcast(tf.constant(value, name=name), root_rank, is_variable=False))


def load_model(keras, wrap_optimizer, optimizer_modules, filepath, custom_optimizers, custom_objects):
    byteps_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ in optimizer_modules
    }

    if custom_optimizers is not None:
        byteps_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        byteps_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=byteps_objects)
