# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Classes for different algorithms of reduction and broadcasting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import six

from tensorflow.python.client import device_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
import tensorflow as tf

from tensorflow.python.distribute.cross_device_ops import CollectiveAllReduce
import tensorflow.python.distribute.cross_device_ops as tf_cross_device_ops


from byteps.tensorflow.ops import broadcast, _push_pull
from byteps.tensorflow import local_rank, size

def check_destinations(destinations):
  """Checks whether `destinations` is not empty.

  Args:
    destinations: a `DistributedValues`, variable, or string object.

  Returns:
    Boolean which is True if `destinations` is not empty.
  """
  # Calling bool() on a ResourceVariable is not allowed.
  if isinstance(destinations,
                (resource_variable_ops.BaseResourceVariable, ops.Tensor)):
    return bool(destinations.device)
  return bool(destinations)


def validate_destinations(destinations):
  """Validates the `destination` is one of expected types."""
  if not isinstance(
      destinations,
      (
          value_lib.DistributedValues,
          resource_variable_ops.BaseResourceVariable,
          ops.Tensor,
          value_lib.AggregatingVariable,
          six.string_types,
          value_lib.TPUMirroredVariable,
          # LogicalDeviceSpec is only used internally, e.g. as a
          # broadcast destination, never supplied by a user.
          value_lib.LogicalDeviceSpec)):
    raise ValueError("destinations must be one of a `DistributedValues` object,"
                     " a tf.Variable object, or a device string.")

  if not check_destinations(destinations):
    raise ValueError("destinations can not be empty")
def reduce_non_distributed_value(reduce_op, device_map, value, destinations):
  """Reduce a non-DistributedValue `value` to `destinations`."""
  if isinstance(value, value_lib.DistributedValues):
    raise ValueError("You are passing a `DistributedValue` to "
                     "`reduce_non_distributed_value`, which is not allowed.")

  # If the same value is present on all replicas then the PerReplica value will
  # be a single value. We also handle the case when `value` is a single value
  # and equal to 0.
  # TODO:(b/138823479): handle the tensor value properly.
  if not tensor_util.is_tensor(value) and value == 0:
    return 0
  # If there is only a single value and the reduce op is MEAN,
  # that value should be on all destinations.
  if reduce_op == reduce_util.ReduceOp.MEAN:
    return value

  validate_destinations(destinations)
  # We do not support a reduce op of SUM if the value is the same across
  # all replicas. We call this as part of assign functions for MirroredVariables
  # and summing up identical values across replicas is not clearly defined.
  if device_map.num_replicas_in_graph != 1:
    raise ValueError("A non-DistributedValues value %s cannot be reduced with "
                     "the given reduce op %s." % (value, reduce_op))
  return simple_broadcast(value, destinations)

def get_device_map_from(destinations):
  if isinstance(destinations, (value_lib.DistributedValues,
                               value_lib.LogicalDeviceSpec)):
    return destinations.device_map, destinations.logical_device
  if isinstance(destinations, six.string_types):
    device = device_util.resolve(destinations)
  else:
    device = destinations.device
  return value_lib.SingleDeviceMap(device), 0

def simple_broadcast(value, destinations, always_mirrored=False):
  """Broadcast `value` to `destinations` using simple copies."""
  device_map, logical_device = get_device_map_from(destinations)
  devices = device_map.logical_to_actual_devices(logical_device)
  if len(devices) == 1 and not always_mirrored:
    return cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        value, devices[0])
  else:
    value_updates = []
    for d in devices:
      value_updates.append(
          cross_device_utils.copy_tensor_or_indexed_slices_to_device(
              value, d))
    return value_lib.regroup(
        device_map, value_updates, wrap_class=value_lib.Mirrored)
def _simple_reduce(per_replica_value, reduce_to_device, accumulation_fn,
                   reduce_op):
  # pylint: disable=g-missing-docstring
  all_values = per_replica_value.values
  if not all_values:
    raise ValueError("`per_replica_value` must be non-empty")
  count = len(all_values)

  if (count == 1 and all_values[0].device == reduce_to_device):
    return all_values[0]

  with ops.device(reduce_to_device):
    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
      reduced = cross_device_utils.aggregate_tensors_or_indexed_slices(
          all_values, accumulation_fn)
      if reduce_op == reduce_util.ReduceOp.MEAN:
        reduced = cross_device_utils.divide_by_n_tensors_or_indexed_slices(
            reduced, count)
      elif reduce_op != reduce_util.ReduceOp.SUM:
        raise ValueError("`reduce_op` must be Reduce.SUM or Reduce.MEAN.")
  return reduced

class CollectiveCommunication(enum.Enum):
  """Communication choices for CollectiveOps.

  * `AUTO`: Default to runtime's automatic choices.
  * `RING`: TensorFlow's ring algorithms for all-reduce and
    all-gather.
  * `NCCL`: Use ncclAllReduce for all-reduce, and ring algorithms for
    all-gather.
  """
  AUTO = "AUTO"
  RING = "RING"
  NCCL = "NCCL"
  # TODO(ayushd): add ncclAllGather implementation.

# TODO(yuefengz): support in-graph collective all-reduce.
class MyCollectiveAllReduce(CollectiveAllReduce):
  """All-reduce cross device ops using collective ops.

  In the between-graph replicated training, it will still do all-reduces across
  all workers and then put results on the right destinations.
  """

  def __init__(self,
               num_workers=1,
               num_gpus_per_worker=0,
               num_packs=1,
               collective_keys=None,
               communication=CollectiveCommunication.AUTO):
    """Initializes the object.

    Args:
      num_workers: number of workers in the between-graph replicated training.
      num_gpus_per_worker: number of GPUs per worker.
      num_packs: gradients will be packed into `num_packs` chunks.
      collective_keys: an optional CollectiveKey object.
      communication: indicates which collective communication to use.
    """
    self._num_workers = num_workers
    self._num_gpus_per_worker = num_gpus_per_worker
    self._num_packs = num_packs
    self._collective_keys = (collective_keys or
                             cross_device_utils.CollectiveKeys())
    self._communication = communication
    super(MyCollectiveAllReduce, self).__init__()

  @property
  def _num_between_graph_workers(self):
    return self._num_workers

  def reduce_implementation(self, reduce_op, per_replica_value, destinations):
    all_reduced = self._batch_all_reduce(reduce_op, [per_replica_value])[0]
    device_map, logical_device = get_device_map_from(destinations)
    devices = device_map.logical_to_actual_devices(logical_device)

    if (isinstance(all_reduced, value_lib.Mirrored) and
        all_reduced.device_map is device_map and
        all_reduced.logical_device == logical_device):
      return all_reduced

    # Convert `all_reduced` to a `Mirrored` object, as a simple and uniform
    # utility to access component for a particular device.
    if not isinstance(all_reduced, value_lib.Mirrored):
      all_reduced = value_lib.Mirrored(
          value_lib.SingleDeviceMap(all_reduced.device), [all_reduced])

    index = []
    with ops.control_dependencies(all_reduced.values):
      for d in devices:
        with ops.device(d):
          if d in all_reduced.devices:
            index.append(array_ops.identity(all_reduced.get(d)))
          else:
            # TODO(josh11b): Once we add support for model parallelism, get the
            # copy from the corresponding replica instead of the primary.
            index.append(array_ops.identity(all_reduced.primary))

    return value_lib.regroup(device_map, index, wrap_class=value_lib.Mirrored)

  def batch_reduce_implementation(self, reduce_op, value_destination_pairs):
    all_devices_match = _all_devices_match(value_destination_pairs)
    if all_devices_match:
      return self._batch_all_reduce(reduce_op,
                                    [v[0] for v in value_destination_pairs])
    else:
      if not all_devices_match:
        logging.log_first_n(
            logging.WARN, "Efficient batch_reduce is not supported if "
            "destinations are different.", 10)

      return [
          self.reduce_implementation(reduce_op, t, destinations=v)
          for t, v in value_destination_pairs
      ]

  def _make_gradient_chunks(self, per_replica_values, num_packs):
    """Make `per_replica_values` into chunks."""
    chunked_by_device = _group_value_by_device(per_replica_values)
    chunked_by_var = list(zip(*chunked_by_device))
    # chunked_by_var is chunked by variables and takes the following format:
    # [((grad0_gpu0, v0_gpu0), (grad0_gpu1, v0_gpu1), (grad0_gpu2, v0_gpu2) ..),
    #  ((grad1_gpu0, v1_gpu0), (grad1_gpu1, v1_gpu1), (grad1_gpu0, v1_gpu2) ..),
    #  ((grad2_gpu0, v2_gpu0), (grad2_gpu1, v2_gpu1), (grad2_gpu0, v2_gpu2) ..),
    #  ...
    # ]

    # No chunking if number of variables is fewer than number of packs.
    if len(chunked_by_var) < num_packs:
      return [chunked_by_var]

    # First n-1 chunks get `chunk_size` grads, last chunk gets leftover grads.
    # This strategy can cause the last chunk to have larger size compared to the
    # first n-1 chunks.  Alternatively, we can increment chunk_size by 1 to get
    # slightly larger first n-1 chunks and smaller last chunk.
    # TODO(ayushd): compare different packing strategies.
    chunk_size = len(chunked_by_var) // num_packs
    leftover_size = len(chunked_by_var) - chunk_size * (num_packs - 1)
    assert leftover_size > 0
    chunked_gv = [
        chunked_by_var[x:x + chunk_size]
        for x in range(0, len(chunked_by_var) - leftover_size, chunk_size)
    ]
    chunked_gv.append(chunked_by_var[-leftover_size:])

    return chunked_gv

  def _batch_all_reduce(self, reduce_op, per_replica_values):
    """All reduce algorithm in a batch."""
    dense_values, dense_indices, sparse_values, sparse_indices = (
        cross_device_utils.split_by_sparsity(per_replica_values))
    if dense_values:
      dense_results = self._do_batch_all_reduce_dense(reduce_op, dense_values)
    else:
      dense_results = []
    if sparse_values:
      sparse_results = self._do_batch_all_reduce_sparse(reduce_op,
                                                        sparse_values)
    else:
      sparse_results = []
    return cross_device_utils.stitch_values(((dense_results, dense_indices),
                                             (sparse_results, sparse_indices)))

  def _do_batch_all_reduce_dense(self, reduce_op, per_replica_values):
    """All-reduce across all workers in a batch."""

    chunked_gv = self._make_gradient_chunks(per_replica_values, self._num_packs)

    batch_size = len(per_replica_values)
    # Pass self._communication to the runtime as a communication hint.
    communication_hint = self._communication.value
    # For now, we use NCCL only when batch_size > 1 and num_packs is 1.
    # TODO(b/132575814): switch to NCCL for all collectives when communication
    # is NCCL.
    if self._communication == CollectiveCommunication.NCCL and (
        batch_size == 1 or self._num_packs != 1):
      communication_hint = CollectiveCommunication.AUTO.value

    logging.log_first_n(
        logging.INFO, "Collective batch_all_reduce: %d all-reduces, "
        "num_workers = %d, communication_hint = %s" % (
            batch_size, self._num_workers, communication_hint), 10)

    reduced_gv_list = []
    for chunk in chunked_gv:
      # By placing all collective ops in a chunk under single name scope, we
      # ensure they will be picked up by the `ScopedAllocator` grappler
      # optimizer and packed into a single all-reduce.
      with ops.name_scope("allreduce"):
        for grad_and_vars in chunk:
          # Gradients for the same variable but from different devices.
          scaled_grads = [g for g, _ in grad_and_vars]
          # collective_reduced = cross_device_utils.build_collective_reduce(
          #     scaled_grads, self._num_workers, self._collective_keys, "Add",
          #     "Id", communication_hint)
          collective_reduced = my_build_collective_reduce(
              scaled_grads, self._num_workers, self._collective_keys, "Add",
              "Id", communication_hint)
          result = []
          for (_, v), g in zip(grad_and_vars, collective_reduced):
            result.append([g, v])
          reduced_gv_list.append(result)

    new_device_grads = [list(x) for x in zip(*reduced_gv_list)]
    return _ungroup_and_make_mirrored(
        new_device_grads,
        per_replica_values[0],
        reduce_op,
        num_between_graph_workers=self._num_workers)

  def _do_batch_all_reduce_sparse(self, reduce_op, per_replica_values):
    """All-reduce IndexedSlices across all workers in a batch."""

    logging.log_first_n(
        logging.INFO, "Collective batch_all_reduce for IndexedSlices: "
        "%d all-reduces, num_workers = %d" %
        (len(per_replica_values), self._num_workers), 10)

    chunked_gv = self._make_gradient_chunks(per_replica_values, self._num_packs)

    reduced_gv_list = []
    for chunk in chunked_gv:
      with ops.name_scope("allreduce"):
        for grad_and_vars in chunk:
          # Gradients for the same variable but from different devices.
          scaled_grads = [g for g, _ in grad_and_vars]

          values = [g.values for g in scaled_grads]
          indices = [g.indices for g in scaled_grads]
          assert len(values) == len(indices)

          # Build two separate allgathers, one for values, the other one for
          # indices.
          gathered_values = cross_device_utils.build_collective_gather(
              values, self._num_workers, self._collective_keys)
          gathered_indices = cross_device_utils.build_collective_gather(
              indices, self._num_workers, self._collective_keys)
          assert len(gathered_values) == len(gathered_indices)

          collective_reduced = []
          for i in range(len(values)):
            reduced = ops.IndexedSlices(
                gathered_values[i],
                gathered_indices[i],
                dense_shape=scaled_grads[i].dense_shape)
            collective_reduced.append(reduced)

          result = []
          for (_, v), g in zip(grad_and_vars, collective_reduced):
            result.append([g, v])
          reduced_gv_list.append(result)

    new_device_grads = [list(x) for x in zip(*reduced_gv_list)]
    return _ungroup_and_make_mirrored(
        new_device_grads,
        per_replica_values[0],
        reduce_op,
        num_between_graph_workers=self._num_workers)


def my_build_collective_reduce(input_tensors,
                            num_workers,
                            collective_keys,
                            reduction_op='Add',
                            unary_op='Id',
                            communication_hint='auto'):
  """Build a subgraph that does one full all-reduce, using the collective Op.

  Args:
    input_tensors: tensors within a single worker graph that are to be reduced
      together; must be one per device.
    num_workers: total number of workers with identical independent graphs that
      will be doing this same reduction.  The reduction will actually include
      the corresponding tensors at all these workers.
    collective_keys: a CollectiveKeys object.
    reduction_op: string naming the reduction op.
    unary_op: string naming the unary final op.
    communication_hint: string providing hint to runtime for choosing collective
      implementation.

  Returns:
    An array of final tensors, one per device, computed by the full reduction.

  Raises:
    ValueError: There must be at least two tensors over all the workers.
  """
  group_size = len(input_tensors) * num_workers
  if group_size < 2:
    return input_tensors
  devices = [t.device for t in input_tensors]
  num_devices = len(devices)
  group_key = collective_keys.get_group_key(devices)
  instance_key = collective_keys.get_op_instance_key()
  subdiv_offsets = [0]  # TODO(tucker): maybe support non-default subdiv spec

  def collective_all_reduce():
    """Call collective allreduce."""
    assert not context.executing_eagerly()
    out_tensors = []
    for d in range(num_devices):
      with ops.device(devices[d]):
        # reduce_op = collective_ops.all_reduce(
        #     input_tensors[d], group_size, group_key, instance_key, reduction_op,
        #     unary_op, subdiv_offsets, communication_hint)
        reduce_op = my_all_reduce(
            input_tensors[d], group_size, group_key, instance_key, reduction_op,
            unary_op, subdiv_offsets, communication_hint)
        out_tensors.append(reduce_op)
    return out_tensors

  if context.executing_eagerly():
    # Collective ops will block unless they are executed concurrently such as in
    # a graph or a defun.
    collective_all_reduce = def_function.function(collective_all_reduce)
  return collective_all_reduce()

def my_all_reduce(t, group_size, group_key, instance_key, merge_op, final_op,
               subdiv_offsets=(0,), communication_hint='auto'):
  """Reduces tensors collectively, across devices.

  Args:
    t: the tensor to be reduced.
    group_size: the total number of tensors to be collectively reduced.
      Each must reside on a different device.  Should be a positive integer.
    group_key: an integer identifying the group of devices.
    instance_key: an integer identifying the participating group of Ops.
    merge_op: string naming the binary Op to be applied to compute each
      partial reduction.
    final_op: string naming the unary Op to be applied to each fully
      reduced value.  Can be 'Id' for no operation.
    subdiv_offsets: a list of integer offsets into the tensor at which each
      independent subdivision should begin.  Use [0] if no subdivision should
      be done.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.

  Returns:
    An Op implementing the distributed reduction.

  Raises:
    ValueError: if any of the input parameter constraints are not met.
  """
  if group_size < 1:
    raise ValueError('Parameter group_size to all_reduce must be at least 1.')
  # return gen_collective_ops.collective_reduce(
  #     t,
  #     group_size=group_size,
  #     group_key=group_key,
  #     instance_key=instance_key,
  #     merge_op=merge_op,
  #     final_op=final_op,
  #     subdiv_offsets=subdiv_offsets,
  #     communication_hint=communication_hint.lower())
  return my_collective_reduce(
      t,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      merge_op=merge_op,
      final_op=final_op,
      subdiv_offsets=subdiv_offsets,
      communication_hint=communication_hint.lower())


def my_collective_reduce(input, group_size, group_key, instance_key, merge_op, final_op, subdiv_offsets, wait_for=[], communication_hint="auto", name=None):
  r"""Mutually reduces multiple tensors of identical type and shape.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: An `int`.
    group_key: An `int`.
    instance_key: An `int`.
    merge_op: A `string` from: `"Min", "Max", "Mul", "Add"`.
    final_op: A `string` from: `"Id", "Div"`.
    subdiv_offsets: A list of `ints`.
    wait_for: An optional list of `ints`. Defaults to `[]`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "CollectiveReduce", name,
        tld.op_callbacks, input, "group_size", group_size, "group_key",
        group_key, "instance_key", instance_key, "merge_op", merge_op,
        "final_op", final_op, "subdiv_offsets", subdiv_offsets, "wait_for",
        wait_for, "communication_hint", communication_hint)
      return _result
    except _core._FallbackException:
      try:
        return collective_reduce_eager_fallback(
            input, group_size=group_size, group_key=group_key,
            instance_key=instance_key, merge_op=merge_op, final_op=final_op,
            subdiv_offsets=subdiv_offsets, wait_for=wait_for,
            communication_hint=communication_hint, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  group_size = _execute.make_int(group_size, "group_size")
  group_key = _execute.make_int(group_key, "group_key")
  instance_key = _execute.make_int(instance_key, "instance_key")
  merge_op = _execute.make_str(merge_op, "merge_op")
  final_op = _execute.make_str(final_op, "final_op")
  if not isinstance(subdiv_offsets, (list, tuple)):
    raise TypeError(
        "Expected list for 'subdiv_offsets' argument to "
        "'collective_reduce' Op, not %r." % subdiv_offsets)
  subdiv_offsets = [_execute.make_int(_i, "subdiv_offsets") for _i in subdiv_offsets]
  if wait_for is None:
    wait_for = []
  if not isinstance(wait_for, (list, tuple)):
    raise TypeError(
        "Expected list for 'wait_for' argument to "
        "'collective_reduce' Op, not %r." % wait_for)
  wait_for = [_execute.make_int(_i, "wait_for") for _i in wait_for]
  if communication_hint is None:
    communication_hint = "auto"
  communication_hint = _execute.make_str(communication_hint, "communication_hint")

  _, _, _op, _outputs = my_new_wrapper_around_byteps_push_pull(
                            input=input, group_size=group_size,
                            group_key=group_key, instance_key=instance_key,
                            merge_op=merge_op, final_op=final_op,
                            subdiv_offsets=subdiv_offsets, wait_for=wait_for,
                            communication_hint=communication_hint, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "group_size",
              _op._get_attr_int("group_size"), "group_key",
              _op._get_attr_int("group_key"), "instance_key",
              _op._get_attr_int("instance_key"), "merge_op",
              _op.get_attr("merge_op"), "final_op", _op.get_attr("final_op"),
              "subdiv_offsets", _op.get_attr("subdiv_offsets"), "wait_for",
              _op.get_attr("wait_for"), "communication_hint",
              _op.get_attr("communication_hint"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollectiveReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

@tf_export("distribute.BytepsAllReduce")
class BytepsAllReduce(tf_cross_device_ops.AllReduceCrossDeviceOps):
  """Reduction using Byteps Push Pull."""

  def __init__(self, num_packs=1):
    """NCCL all-reduce implementation of CrossDeviceOps.

    It uses Nvidia NCCL for all-reduce. Before performing all-reduce, tensors
    will be repacked or aggregated for more efficient cross-device
    transportation.

    Args:
      num_packs: values will be packed in this many splits.  `num_packs` should
        be greater than or equals 0. When it is zero, no packing will be done.

    Raises:
      ValueError if `num_packs` is negative.
    """
    if num_packs < 0:
      raise ValueError(
          "NCCL all-reduce requires num_packs >= 0, but {} is specified".format(
              num_packs))
    super(BytepsAllReduce, self).__init__(
        all_reduce_alg="nccl", num_packs=num_packs)
    # self._simple_cross_replica_ops = ReductionToOneDevice()
    self._simple_cross_replica_ops = BytepsCrossDeviceOps()


class BytepsCrossDeviceOps(tf_cross_device_ops.CrossDeviceOps):
  def __init__(self):
    self.accumulation_fn = math_ops.add_n
  def reduce_implementation(self, reduce_op, per_replica_value, destinations):
    if tf_cross_device_ops.check_destinations(destinations):
      devices = tf_cross_device_ops.get_devices_from(destinations)
    else:
      devices = tf_cross_device_ops.get_devices_from(per_replica_value)
    reduce_to_device = devices[0]
    logging.log_first_n(
        logging.INFO, "Using byteps push pull to aggregate values", 1)
    reduced = _simple_reduce(per_replica_value, reduce_to_device,
                             self.accumulation_fn, reduce_op)
    if size() > 1:
        reduced = _push_pull(reduced)
    return reduced
