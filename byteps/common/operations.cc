// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "operations.h"
#if BYTEPS_BUILDING_CUDA == 1
#include <cuda_runtime.h>
#endif

#include <unistd.h>
#include <cstring>
#include <memory>
#include <thread>
#include <numa.h>

#include "compressor/compressor.h"
#include "compressor/compressor_registry.h"
#include "compressor/utils.h"
#include "core_loops.h"
#include "global.h"
#include "logging.h"
#include "operations.h"

namespace byteps {
namespace common {

extern "C" {

void byteps_init() {
  byteps_lazy_init();
  BytePSGlobal::GetOrInitPS();
  BPS_LOG(INFO) << "byteps_init() DONE " << BytePSGlobal::GetRank();
}

void byteps_lazy_init() {
  BytePSGlobal::Init();

  // The order of func does not matter
  std::vector<LoopFunction> func;

  // Push & Pull in distributed mode
  if (BytePSGlobal::IsDistributed()) {
    func.push_back(RecvLoop);
    func.push_back(SendLoop);
    func.push_back(P2PPullLoop);
    func.push_back(P2PPullResponseLoop);
    func.push_back(P2PGroupCopyHost2DeviceLoop);
    func.push_back(P2PAckLoop);

    if (BytePSGlobal::IsRootDevice()) {
      func.push_back(PullLoop);
      func.push_back(DecompressLoop);
    }
  }

  // Cross-PCIe-switch reduce
  if (BytePSGlobal::IsCrossPcieSwitch()) {
    func.push_back(PcieReduceLoop);
  }

  // Copy between GPU and CPU
  if (BytePSGlobal::IsCrossPcieSwitch() || BytePSGlobal::IsDistributed()) {
    func.push_back(CopyDevice2HostLoop);
    if (BytePSGlobal::IsRootDevice()) {
      // PUSH can be a real push in distributed mode
      // Or a dummy barrier in cross-pcie-switch mode
      func.push_back(PushLoop);
      func.push_back(CompressLoop);
      func.push_back(RootCopyHost2DeviceLoop);
    } else {
      func.push_back(CoordinatePushLoop);
      func.push_back(NonRootCopyHost2DeviceLoop);
      // DO_COPYH2D handling moved to commsocket listen thread
      // func.push_back(NonRootCopyListenLoop);
    }
  }

  // Per-PCIe-switch NCCL calls
#if BYTEPS_BUILDING_CUDA == 1
  func.push_back(SyncNcclLoop);
  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    func.push_back(RootNcclLoop);
  } else {
    func.push_back(CoordinateReduceLoop);
    func.push_back(CoordinateBroadcastLoop);
    func.push_back(NonRootNcclLoop);
  }
#endif
  if (BytePSGlobal::IsRootDevice()) {
    func.push_back(CpuCopyLoop);
    func.push_back(CpuReduceLoop);
    func.push_back(CpuBcastLoop);
    func.push_back(CpuBcastFinishLoop);
  } else {
    func.push_back(CpuCopyLoop);
    func.push_back(CpuReduceLoop);
    func.push_back(CpuBcastLoop);
  }
  BytePSGlobal::Start(func);
  return;
}

void byteps_shutdown() {
  BytePSGlobal::Shutdown();
  BPS_LOG(DEBUG) << "BytePS has been completely shutdown now";
  return;
}

void byteps_resume(int num_workers, int num_servers) {
  // set ps, worker numbers
  BPS_LOG(DEBUG) << "Resume worker number: " << num_workers
                 << "DMLC_NUM_WORKER: " << getenv("DMLC_NUM_WORKER");
  BPS_LOG(DEBUG) << "Resume server number: " << num_workers
                 << "DMLC_NUM_SERVER: " << getenv("DMLC_NUM_SERVER");
  BPS_LOG(DEBUG) << "Start resuming BytePS";

  BytePSGlobal::SetResumingFlag(true);
  byteps_init();

  // redeclare tensor with original order
  BytePSGlobal::ReDeclareTensor();
  BytePSGlobal::SetResumingFlag(false);

  BPS_LOG(INFO) << "BytePS has been resumed now";
}

void byteps_suspend() {
  BPS_LOG(DEBUG) << "Start suspending BytePS";
  BytePSGlobal::Shutdown();
  BPS_LOG(INFO) << "BytePS has been suspended now";
  return;
}

int byteps_rank() { return BytePSGlobal::GetRank(); }

int byteps_local_rank() { return BytePSGlobal::GetLocalRank(); }

int byteps_size() { return BytePSGlobal::GetSize(); }

int byteps_local_size() { return BytePSGlobal::GetLocalSize(); }

uint64_t byteps_session_id(const char* name) { return BytePSGlobal::GetSessionId(std::string(name)); }

void byteps_mark_done(const char* name) { BytePSGlobal::MarkDone(std::string(name)); return; }

uint32_t byteps_session_size() { return BytePSGlobal::GetSessionSize(); }

void byteps_get_telemetry_size(int32_t* size) {
  size[0] = Telemetry::size();
}

void byteps_get_telemetry_data(const char** names, float* mean, float* stdev,
                               int* count, int* actual_size, int max_size) {
  Telemetry::GetData(names, mean, stdev, count, actual_size, max_size);
}

}  // extern "C"

Status CheckInitialized() { return BytePSGlobal::CheckInit(); }

void PartitionTensor(
    std::shared_ptr<TensorTableEntry> entry,
    std::vector<std::shared_ptr<TensorTableEntry>> &partitions) {
  BPS_CHECK(entry->counter_ptr)
      << entry->tensor_name << " counter pointer is null";
  size_t size = entry->tensor ? entry->tensor->size() : entry->output->size();
  size_t bound = BytePSGlobal::GetPartitionBound();
  size_t accumulated = 0;
  int i = 0;

  while (accumulated < size) {
    std::shared_ptr<TensorTableEntry> e(new TensorTableEntry(entry->priority, entry->version,
                                                             entry->ready_event, entry->callback,
                                                             entry->device, entry->queue_list));
    // will assign the key later, so don't do it now
    // e->key = entry->key;
    e->context = entry->context;
    e->tensor_name = entry->tensor_name + std::string("_") + std::to_string(i);
    e->len = ((size - accumulated) > bound) ? bound : (size - accumulated);
    // Short-cut for P2P ops
    if (entry->context->op_type != PUSH_PULL_OP) {
      auto ctx = entry->context;
      if (BytePSGlobal::IsSkipD2H() && entry->tensor && entry->tensor->data()) {
        e->cpubuff = ((char*)entry->tensor->data()) + accumulated;
      } else {
        e->cpubuff = ctx->cpubuff_list.at(i);
      }
    } else {
      e->cpubuff = entry->cpubuff;
    }
    e->offset = accumulated;
    e->gpu_ptr = entry->gpu_ptr;
    e->pcie_cpubuff = entry->pcie_cpubuff;
    e->numa_cpubuff = entry->numa_cpubuff;
    e->tensor = entry->tensor;
    e->output = entry->output;

    e->counter_ptr = entry->counter_ptr;
    e->total_partnum = entry->total_partnum;
    e->reduce_op = entry->reduce_op;
    if (!entry->context->compressor_list.empty()) {
      e->compressor = entry->context->compressor_list[i];
    }
    accumulated += e->len;
    ++i;

    partitions.push_back(e);
  }
}

Status PrepareAlltoallTensor(TensorShape shape,
  const std::vector<int32_t>& tensor_key,
  const std::vector<int32_t>& split_list,
  const std::vector<int32_t>& recv_split_list,
  std::string& name,
  std::vector<int32_t>* split_indices_list,
  std::vector<int32_t>* recv_split_indices_list,
  int32_t* dim0_in, int32_t* dim0_out,
  std::string* session_name, bool* initialized) {
  // calculate the stride based on shape[1:]
  int64_t stride = 1;
  for (int i = 1; i < shape.dims(); ++i) {
    stride *= shape.dim_size(i);
  }
  // calculate split indices
  for (size_t i = 0; i < split_list.size(); ++i) {
    int32_t split_i = split_list.at(i);
    *dim0_in += split_i;
    // the split tensor is based on axis 0, hence scale it by stride
    split_indices_list->push_back(split_i * stride);
    if (split_i < 0) {
      std::string reason = name + ": invalid split[" + std::to_string(i) + "]="+ std::to_string(split_list[i]);
      return Status::InvalidArgument(reason);
    }
  }
  // sanity check: sum(split) == shape[0]
  auto expected_dim0 = shape.dim_size(0);
  if (*dim0_in != expected_dim0) {
    std::string reason;
    for (size_t i = 0; i < split_list.size(); ++i) {
      reason += std::to_string(split_list.at(i)) + ",";
    }
    reason = name + ": invalid split. tensor.shape[0]=" + std::to_string(expected_dim0) + ". split=" + reason;
    return Status::InvalidArgument(reason);
  }
  // calculate recv_split indices
  for (size_t i = 0; i < recv_split_list.size(); ++i) {
    int32_t recv_split_i = recv_split_list.at(i);
    if (recv_split_i < 0) {
      std::string reason = name + ": invalid recv_split[" + std::to_string(i) + "]=" + std::to_string(recv_split_i);
      return Status::InvalidArgument(reason);
    }
    *dim0_out += recv_split_i;
    recv_split_indices_list->push_back(recv_split_i * stride);
  }

  // naming and declarations
  const int my_rank = common::byteps_rank();
  // Add session_id prefix to node_name
  std::string name_send = name + "_alltoall_send_" + std::to_string(my_rank) + "_recv_0";
  int session_id = common::byteps_session_id(name_send.c_str());
  int session_size = common::byteps_session_size();
  std::string session_prefix = "session_" + std::to_string(session_id % session_size) + "_";
  std::string session_name_send = session_prefix + name_send;
  *session_name = session_prefix + name;

  for (int j = 0; j < tensor_key.size(); ++j) {
    int num_ranks = split_list.size();
    std::string sess_prefix = "session_" + std::to_string(j) + "_";
    for (size_t i = 0; i < num_ranks; ++i) {
      std::string nsend = sess_prefix + name + "_alltoall_send_"
                        + std::to_string(my_rank) + "_recv_" + std::to_string(i);
      std::string nrecv = sess_prefix + name + "_alltoall_send_"
                        + std::to_string(i) + "_recv_" + std::to_string(my_rank);
      common::IsTensorDeclaredP2P(nsend, my_rank, i, tensor_key.at(j));
      common::IsTensorDeclaredP2P(nrecv, i, my_rank, tensor_key.at(j));
    }
  }
  auto& bps_context = common::GetContextFromName(session_name_send);
  *initialized = bps_context.initialized;
  return Status::OK();
}

Status EnqueueAlltoAllTensor(std::string& name,
                             std::shared_ptr<Tensor> input,
                             std::vector<std::shared_ptr<Tensor>>& group_inputs,
                             std::shared_ptr<Tensor> output,
                             std::vector<std::shared_ptr<Tensor>>& group_outputs,
                             std::shared_ptr<Tensor> size_output,
                             std::shared_ptr<ReadyEvent> ready_event,
                             const int input_device, const int output_device,
                             const int priority, const int version,
                             StatusCallback callback,
                             const std::vector<int>& send_begin, // begin offsets for send
                             const std::vector<int>& recv_begin, // begin offsets for recv
                             std::atomic_int* counter_ptr,
                             bool output_size_unknown) {
  if (BytePSGlobal::ShouldShutdown()) {
    return Status::OK();
  }

  // ========== COMMON INFO =========
  // send_begin always starts with a zero
  int num_ranks = send_begin.size() - 1;
  auto dtype = input ? input->dtype() : group_inputs[0]->dtype();
  auto unit_size = getDataTypeLength(dtype);
  const int my_rank = common::byteps_rank();
  int next_rank = (my_rank + 1) % num_ranks;
  // naming
  std::string name_prefix = name + "_alltoall_send_";
  std::string my_rank_str = std::to_string(my_rank);
  // format: xx_alltoall_send_yy_recv_me
  std::string recv_name_suffix = "_recv_" + my_rank_str;
  // format: xx_alltoall_send_me_recv_zz
  std::string send_name_prefix = name_prefix + my_rank_str + "_recv_";
  std::string reference_send_name = send_name_prefix + std::to_string(next_rank);
  std::string reference_recv_name = name_prefix + std::to_string(next_rank) + recv_name_suffix;
  // track the tasks and determine if we need to update counter_ptr
  // if the output_size is unknown, there will definitely be ps-lite send/recv operations
  bool has_send = output_size_unknown;
  bool has_recv = output_size_unknown;

  // ========= SEND TASKS ==========
  std::vector<QueueType> send_q = {SEND};
  auto send_task = new P2PTensorTableEntry(priority, version, ready_event, callback,
                                           input_device, send_q, output_device);
  // the accumulated offset list always starts with 0
  send_task->offset = 0;
  send_task->offset_list.push_back(0);
  // send to all ranks
  bool recv_on_gpu = (output_device != CPU_DEVICE_ID);
  for (size_t i = 0; i < num_ranks; ++i) {
    // Note: shuffle is done inside core_loops
    int size = unit_size * (send_begin[i+1] - send_begin[i]);
    send_task->offset_list.push_back(send_begin[i + 1] * unit_size);
    std::string name_i = send_name_prefix + std::to_string(i);
    auto& byteps_context = common::GetContextFromName(name_i);
    common::InitTensorP2P(byteps_context, size, dtype, nullptr, my_rank, i, recv_on_gpu);
    // XXX: use pcie_cpubuff as container of the list of aligned memory buffs
    send_task->pcie_cpubuff.push_back(byteps_context.cpubuff_list[0]);
    send_task->key_list.push_back(byteps_context.key_list[0]);
    // when output size is known, check if we need to update `has_send
    // we check the case for send-recv with myself, or nothing to send
    if (i != my_rank && size != 0) {
      has_send = true;
      // avoid init shared_ptr if has_send == false
      if (!send_task->counter_a2a.get()) { 
        send_task->counter_a2a = std::make_shared<std::atomic_int>(0);
      }
      send_task->counter_a2a.get()->fetch_add(1);
    }
  }
  if (has_send) {
    send_task->tensor_name = reference_send_name;
    // use the reference send task's context
    auto& byteps_context = common::GetContextFromName(reference_send_name);
    send_task->context = &byteps_context;
    send_task->tensor = input;
    send_task->group_tensors.assign(group_inputs.begin(), group_inputs.end());
    if (!output_size_unknown) {
      send_task->output = output;
      send_task->group_outputs.assign(group_outputs.begin(), group_outputs.end());
    }
    // TODO: remove counter_ptr. It is not necessary.
    send_task->counter_ptr = std::make_shared<std::atomic_int>(0);
    send_task->total_partnum = 1;
    BytePSGlobal::GetScheduledQueue(SEND)->addTask(send_task);
  } else {
    // nothing to send
    --(*counter_ptr);
    delete send_task;
  }

  // ========= RECV TASKS  ==========
  // reference tensor_key for group_recv
  std::vector<QueueType> recv_q = {RECV};
  if (output_size_unknown) {
    recv_q = {P2P_GROUP_COPYH2D};
  }
  uint64_t tensor_key;
  int total_partnum = 0;
  // create the shared recv tasks
  auto recv_task = new P2PTensorTableEntry(priority, version, ready_event, callback,
                                           input_device, recv_q, output_device);
  recv_task->offset = 0;
  recv_task->tensor = input;
  recv_task->group_tensors.assign(group_inputs.begin(), group_inputs.end());
  recv_task->output = output;
  recv_task->group_outputs.assign(group_outputs.begin(), group_outputs.end());
  recv_task->aux_output = size_output;

  // the accumulated offset list always starts with 0
  recv_task->offset_list.push_back(0);
  std::vector<P2PTensorTableEntry*> recv_tasks;
  for (size_t i = 0; i < num_ranks; ++i) {
    // check the case for send-recv with myself, or nothing to send
    int size = unit_size * (recv_begin[i + 1] - recv_begin[i]);
    std::string name_i = name_prefix + std::to_string(i) + recv_name_suffix;
    auto& byteps_context = common::GetContextFromName(name_i);
    common::InitTensorP2P(byteps_context, size, dtype, nullptr, i, my_rank, recv_on_gpu);
    tensor_key = (byteps_context.key_list[0] << 32) >> 48;
    recv_task->offset_list.push_back(recv_begin[i + 1] * unit_size);
    if (output_size_unknown) {
      total_partnum = 1;
      has_recv = true;
    } else {
      if (size != 0) {
        total_partnum++;
        has_recv = true;
        auto partition = new P2PTensorTableEntry(priority, version, ready_event, callback,
                                                 input_device, recv_q, output_device);
        *partition = *recv_task;
        partition->tensor_name = name_i;
        partition->context = &byteps_context;
        partition->key = byteps_context.key_list[0];
        partition->len = size;
        partition->counter_ptr = std::make_shared<std::atomic_int>(0);
        recv_tasks.push_back(partition);
        if (i == my_rank) {
          // for the self send-recv case, we need to know the offset of the input tensor
          partition->offset = send_begin[i] * unit_size;
          BytePSGlobal::GetP2PCopyTable()->AddReadyCount(partition->key);
        }
      } else {
        auto count = --(*counter_ptr);
        if (count == 0) callback(Status::OK());
      }
    }
  }
  if (has_recv) {
    recv_task->total_partnum = total_partnum;
    if (output_size_unknown) {
      // use the reference recv task's context
      auto& byteps_context = common::GetContextFromName(reference_recv_name);
      recv_task->tensor_name = reference_recv_name;
      recv_task->context = &byteps_context;
      recv_task->key = tensor_key;
      recv_task->counter_ptr = std::make_shared<std::atomic_int>(0);
      BytePSGlobal::GetScheduledQueue(recv_q.at(0))->addTask(recv_task);
    } else {
      for (auto partition : recv_tasks) {
        partition->total_partnum = 1;
        partition->offset_list = recv_task->offset_list;
        BytePSGlobal::GetScheduledQueue(recv_q.at(0))->addTask(partition);
      }
      delete recv_task;
    }
  } else {
    // nothing to recv
    delete recv_task;
  }
  BPS_LOG(TRACE) << "EnqueueAlltoAllTensor finished: " << name
                << ", rank=" << BytePSGlobal::GetRank()
                << ", has_send=" << has_send
                << ", has_recv=" << has_recv;
  return Status::OK();
}

Status EnqueueAlltoAllTensorPullImpl(
                             std::string& name,
                             std::shared_ptr<Tensor> input,
                             std::vector<std::shared_ptr<Tensor>>& group_inputs,
                             std::shared_ptr<Tensor> output,
                             std::vector<std::shared_ptr<Tensor>>& group_outputs,
                             std::shared_ptr<Tensor> size_output,
                             std::shared_ptr<ReadyEvent> ready_event,
                             const int input_device, const int output_device,
                             const int priority, const int version,
                             StatusCallback callback,
                             const std::vector<int>& send_begin, // begin offsets for send
                             const std::vector<int>& recv_begin, // begin offsets for recv
                             std::atomic_int* counter_ptr,
                             bool output_size_unknown) {
  if (BytePSGlobal::ShouldShutdown()) {
    return Status::OK();
  }
  BPS_CHECK(!output_size_unknown);
  // ========== COMMON INFO =========
  // send_begin always starts with a zero
  int num_ranks = send_begin.size() - 1;
  auto dtype = input ? input->dtype() : group_inputs[0]->dtype();
  auto unit_size = getDataTypeLength(dtype);
  const int my_rank = common::byteps_rank();
  int next_rank = (my_rank + 1) % num_ranks;
  // naming
  std::string name_prefix = name + "_alltoall_send_";
  std::string my_rank_str = std::to_string(my_rank);
  std::string recv_name_suffix = "_recv_" + my_rank_str; // xx_alltoall_send_yy_recv_me
  std::string send_name_prefix = name_prefix + my_rank_str + "_recv_"; // xx_alltoall_send_me_recv_zz
  std::string reference_send_name = send_name_prefix + std::to_string(next_rank);
  // track the tasks and determine if we need to update counter_ptr
  // if the output_size is unknown, there will definitely be ps-lite send/recv operations
  bool has_request = output_size_unknown;
  bool has_response = output_size_unknown;

  // ========= REQUEST TASKS ==========
  std::vector<QueueType> request_q = {P2P_PULL};
  auto request_task = new P2PTensorTableEntry(priority, version, ready_event,
                                              callback, input_device, request_q, output_device);
  // the accumulated offset list always starts with 0
  request_task->offset_list.push_back(0);
  // send to all ranks
  bool recv_on_gpu = (output_device != CPU_DEVICE_ID);
  for (size_t i = 0; i < num_ranks; ++i) {
    // Note: shuffle is done inside core_loops
    int size = unit_size * (recv_begin[i+1] - recv_begin[i]);
    request_task->offset_list.push_back(recv_begin[i+1] * unit_size);
    std::string name_i = send_name_prefix + std::to_string(i);
    auto& byteps_context = common::GetContextFromName(name_i);
    common::InitTensorP2P(byteps_context, size, dtype, nullptr, my_rank, i, recv_on_gpu);
    // XXX: use pcie_cpubuff as container of the list of aligned memory buffs
    request_task->pcie_cpubuff.push_back(byteps_context.cpubuff_list[0]);
    request_task->key_list.push_back(byteps_context.key_list[0]);
    // when output size is known, check if we need to update `has_request
    // we check the case for send-recv with myself, or nothing to send
    if (i != my_rank && size != 0) {
      has_request = true;
      // avoid init shared_ptr if has_request == false
      if (!request_task->counter_a2a.get()) { 
        request_task->counter_a2a = std::make_shared<std::atomic_int>(0);
      }
      request_task->counter_a2a.get()->fetch_add(1);
    }
  }
  if (has_request) {
    request_task->tensor_name = reference_send_name;
    // use the reference send task's context
    auto& byteps_context = common::GetContextFromName(reference_send_name);
    request_task->context = &byteps_context;
    request_task->tensor = input;
    request_task->group_tensors.assign(group_inputs.begin(), group_inputs.end());
    request_task->output = output;
    request_task->group_outputs.assign(group_outputs.begin(), group_outputs.end());
    // TODO: remove counter_ptr. It is not necessary.
    request_task->counter_ptr = std::make_shared<std::atomic_int>(0);
    request_task->total_partnum = 1;
    BytePSGlobal::GetScheduledQueue(P2P_PULL)->addTask(request_task);
  } else {
    // nothing to send
    --(*counter_ptr);
    delete request_task;
  }

  // ========= RESPONSE TASKS  ==========
  int total_partnum = 0;
  // create the shared recv tasks
  std::vector<QueueType> resp_q = {P2P_PULL_RESPONSE, P2P_WAIT_ACK};
  if (BytePSGlobal::IsP2PAckDisabled()) {
    resp_q = {P2P_PULL_RESPONSE};
  }
  auto response_task = new P2PTensorTableEntry(priority, version, ready_event, callback,
                                               input_device, resp_q, output_device);
  response_task->aux_output = size_output;
  response_task->group_tensors.assign(group_inputs.begin(), group_inputs.end());
  response_task->group_outputs.assign(group_outputs.begin(), group_outputs.end());

  // the accumulated offset list always starts with 0
  response_task->offset_list.push_back(0);
  std::vector<P2PTensorTableEntry*> tasks;
  bool is_group = (group_inputs.size() > 0);
  for (size_t i = 0; i < num_ranks; ++i) {
    // check the case for send-recv with myself, or nothing to send
    int size = unit_size * (send_begin[i+1] - send_begin[i]);
    std::string name_i = name_prefix + std::to_string(i) + recv_name_suffix;
    auto& byteps_context = common::GetContextFromName(name_i);
    common::InitTensorP2P(byteps_context, size, dtype, nullptr, i, my_rank, recv_on_gpu);
    response_task->offset_list.push_back(send_begin[i+1] * unit_size);
    if (size != 0) {
      total_partnum++;
      has_response = true;
      auto partition = new P2PTensorTableEntry(priority, version, ready_event, callback,
                                               input_device, resp_q, output_device);
      *partition = *response_task;
      partition->tensor = is_group
                        ? group_inputs[i]
                        : input;
      partition->offset = is_group
                        ? 0
                        : response_task->offset_list[i];
      partition->tensor_name = name_i;
      partition->context = &byteps_context;
      partition->key = byteps_context.key_list[0];
      partition->len = size;
      partition->counter_ptr = std::make_shared<std::atomic_int>(0);
      tasks.push_back(partition);
      if (i == my_rank) {
        partition->output = is_group
                        ? group_outputs[i]
                        : output;
        // this offset is for output tensor, so use recv_begin instead of send_begin
        partition->offset = is_group 
                        ? 0
                        : recv_begin[i] * unit_size;
        BytePSGlobal::GetP2PPullResponseTable()->AddReadyCount(partition->key);
      }
    } else {
      auto count = --(*counter_ptr);
      if (count == 0) callback(Status::OK());
    }
  }
  if (has_response) {
    response_task->total_partnum = total_partnum;
    for (auto& partition : tasks) {
      partition->total_partnum = 1;
      partition->offset_list = response_task->offset_list;
      BytePSGlobal::GetScheduledQueue(P2P_PULL_RESPONSE)->addTask(partition);
    }
  } 
  delete response_task;
  BPS_LOG(TRACE) << "EnqueueAlltoAllTensorPullImpl finished: " << name
                << ", rank=" << BytePSGlobal::GetRank()
                << ", has_request=" << has_request
                << ", has_response=" << has_response;
  return Status::OK();
}

Status EnqueueTensor(BPSContext &context, std::shared_ptr<Tensor> input,
                     std::shared_ptr<Tensor> output,
                     std::shared_ptr<ReadyEvent> ready_event, const int device,
                     const int priority, const int version,
                     StatusCallback callback,
                     std::shared_ptr<std::vector<QueueType>> queue_list,
                     ReduceOp op) {
  if (BytePSGlobal::ShouldShutdown()) {
    return Status::OK();
  }

  auto &name = context.tensor_name;
  if (input && output && context.op_type == PUSH_PULL_OP) {
    BPS_CHECK_EQ(input->size(), output->size())
      << name << " output tensor size does not match";
  }

  // add queue
  if (BytePSGlobal::IsRootDevice() && !context.compressor_list.empty()) {
    auto it = std::find(queue_list->begin(), queue_list->end(), PUSH);
    it = queue_list->insert(it, COMPRESS);  // before PUSH
    it = std::find(queue_list->begin(), queue_list->end(), PULL);
    queue_list->insert(it + 1, DECOMPRESS);  // after PULL
  }

  std::shared_ptr<TensorTableEntry> e(new TensorTableEntry(priority, version,
                                                           ready_event, callback,
                                                           device, *queue_list));
  e->tensor_name = name;
  e->context = &context;
  // Note: for the send-recv case, one may have null input or output
  e->tensor = input;
  e->output = output;
  e->reduce_op = op;

  // send/recv ops do not need gpu_ptr
  if (device == CPU_DEVICE_ID && context.op_type == PUSH_PULL_OP) {
#if BYTEPS_BUILDING_CUDA == 1
    cudaError_t err = cudaHostRegister(const_cast<void *>(input->data()),
                                       input->size(), cudaHostRegisterMapped);
    if (err == cudaSuccess) {
      BPS_LOG(DEBUG) << name
                     << " cpu address has changed, so it is pinned again.";
    }
    BPS_CHECK(input->data() != nullptr);
    CUDA_CALL(cudaHostGetDevicePointer(&(context.gpu_ptr), const_cast<void*>(input->data()), 0));
#else
    context.gpu_ptr = const_cast<void*>(input->data());
#endif
  }

  e->cpubuff = context.cpubuff;
  e->gpu_ptr = context.gpu_ptr;
  e->pcie_cpubuff = context.pcie_cpubuff;
  e->numa_cpubuff = context.numa_cpubuff;
  e->counter_ptr = std::make_shared<std::atomic_int>(0);
  e->total_partnum = context.key_list.size();

  std::vector<std::shared_ptr<TensorTableEntry>> partitions;
  PartitionTensor(e, partitions);
  BPS_CHECK_EQ(context.key_list.size(), partitions.size())
      << name << ": " << context.key_list.size() << ", " << partitions.size();


  if (e->queue_list.size() == 0) {
    BPS_CHECK(e->tensor_name != "");
    BPS_LOG(TRACE) << e->tensor_name << ", device=" << e->device
                   << " has no queue_list assigned, skipped";
    e->callback(Status::OK());
    return Status::OK();
  }

  // add for profiling
  if (context.profile_flag) {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

    BPSCommTime *ret = new BPSCommTime;
    ret->start_t = (long long)(us.count());
    context.comm_time.push(ret);
  }

  unsigned int accumulated = 0;
  for (size_t i = 0; i < partitions.size(); ++i) {
    auto task = partitions[i];
    task->key = context.key_list[i];  // assign the key now
    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "EnqueueTensor: " << (task->tensor_name)
                   << ", key=" << (task->key) << ", offset=" << (task->offset)
                   << ", len=" << (task->len) << ", device=" << (task->device)
                   << ", local_rank=" << BytePSGlobal::GetLocalRank();

    BytePSGlobal::GetScheduledQueue(e->queue_list[0])->addTask(task);
    accumulated += task->len;
  }

  auto tensor = (e->tensor ? e->tensor : e->output);
  BPS_CHECK(tensor);
  BPS_CHECK_EQ(accumulated, tensor->size())
      << "accumulated partition size not equal to original tensor size";

  BPS_LOG(TRACE) << "EnqueueTensor finished: " << name
                 << ", rank=" << BytePSGlobal::GetLocalRank();

  return Status::OK();
}

void InitTensorP2P(BPSContext &context, size_t size, int dtype, void *cpubuff,
                   int sender, int receiver, bool recv_on_gpu) {
  auto bound = BytePSGlobal::GetP2PPartitionBound();
  std::lock_guard<std::mutex> lock(context.init_mutex);
  // XXX: in alltoall, the size might be 0 when first seen.
  // Nonetheless, we initialize the memory buff for it
  size = size == 0 ? 1 : size;
  if (context.initialized) {
    // XXX we assume the number of partitions do not change
    int num_partitions = (size + bound - 1) / bound;
    BPS_CHECK_EQ(context.key_list.size(), num_partitions)
      << "Unexpected tensor partition count: "
      << num_partitions << " v.s. " << context.key_list.size();
    return;
  }
  if (sender == -1) {
    sender = BytePSGlobal::GetRank();
  }
  if (receiver == -1) {
    receiver = BytePSGlobal::GetRank();
  }
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  // Get metadata
  auto &name = context.tensor_name;
  context.buff_len = size;
  context.sender = sender;
  context.receiver = receiver;
  size_t accumulated = 0;

  // Add for timeline
  BytePSGlobal::SetProfileFlag(&context);
  context.local_rank = BytePSGlobal::GetLocalRank();

  // Total key space is [0, 2^64 - 1]
  // It will be divided to N PS servers, for now we assume N <= 2^16
  // Then we have 2^48 key space left.
  // Top 16 bits out of the 48 bits encodes the sender rank
  // Mid 16 bits out of the 48 bits encodes the tensor id
  // The next 6 bits encodes request types (pushpull, send, etc)
  // The last 10 bits encodes the partition id
  // Therefore, we support up to 2^16 tensors, and up to 2^10 partitions per tensor
  ps::Key start_key = (uint64_t) sender << 32;
  start_key += ((uint64_t) context.declared_key) << 16;
  start_key += (uint32_t) P2P_OP << 10;
  while (accumulated < size) {
    context.key_list.push_back(start_key++);
    accumulated +=
        ((size - accumulated) > bound) ? bound : (size - accumulated);
  }

  BPS_LOG(DEBUG) << name << " partitioned to " << context.key_list.size()
                 << " part(s)"
                 << ", total_len=" << size << ", key_range=["
                 << context.key_list.front() << ", " << context.key_list.back()
                 << "] worker_id=" << BytePSGlobal::GetWorkerID()
		 << ", sender=" << sender << ", receiver=" << receiver;

  auto key_list = context.key_list;

  // For now, we do not implement partitioning for P2P operations
  BPS_CHECK_EQ(key_list.size(), 1)
      << name << ", size=" << size << ", buff_size=" << bound
      << ", please consider increasing BYTEPS_P2P_PARTITION_BYTES";

  BPS_LOG(TRACE) << "Begin init " << name << ", size=" << size
                 << ", parts=" << key_list.size();

  // P2P operations do not need to register tensor with CUDA for NCCL
  context.gpu_ptr = nullptr;

  // We always allocate our own cpu buffer
  // use the first key in key_list as the index
  auto shm_obj = BytePSGlobal::GetSharedMemoryObj();
  CHECK(!BytePSGlobal::IsCrossPcieSwitch());
  // use a different prefix for p2p tensors
  std::string wid = "_" + std::to_string(BytePSGlobal::GetWorkerID()) + "_";
  std::string shm_name = std::string("BytePS_P2P_ShM_") + BytePSGlobal::GetUUID() + wid;
  accumulated = 0;
  context.cpubuff = nullptr;
  auto my_rank = BytePSGlobal::GetRank();
  CHECK(BytePSGlobal::IsDistributed());
  auto ps = BytePSGlobal::GetOrInitPS();
  for (auto k : key_list) {
    int len = ((size - accumulated) > bound) ? bound : (size - accumulated);
    // XXX: We assume the number of partitions do not change
    // When encoding for the first time, declare len = bound
    auto pskv = BytePSGlobal::EncodeP2PKey(k, bound, receiver);
    // the shared memory is always created at partition size
    void* buff = nullptr;
    if (sender == my_rank) {
      buff = shm_obj->openSharedMemory(shm_name, pskv.keys[0], bound);
      context.cpubuff_list.emplace_back(buff);
      // false means not to delete data when SArray is deleted
      ps::SArray<char> vals((char*) buff, bound, false);
      DeviceType device = recv_on_gpu ? GPU : CPU;
      int cmd = server::GetCommandType(server::RequestType::kDefaultSend, dtype, device);
      if (!BytePSGlobal::IsAlltoallUsePull()) {
        // blocking push, also as a global barrirer
        ps->Wait(ps->ZPush(pskv.keys, vals, pskv.lens, cmd));
      }
    } else {
      // no need to create the cpubuff as a receiver
      context.cpubuff_list.emplace_back(buff);
    }
    accumulated += len;
  }
  BPS_CHECK_EQ(accumulated, size);

  BPS_LOG(TRACE) << name << ": open shared memory size " << size;

  context.initialized = true;

  BPS_LOG(TRACE) << "Finish Init " << name << ", size=" << size
                 << ", parts=" << key_list.size();
}

void InitTensor(BPSContext &context, size_t size, int dtype, void *cpubuff) {
  std::lock_guard<std::mutex> lock(context.init_mutex);
  if (context.initialized) {
    return;
  }
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));

  BPS_CHECK_GT(size, 0) << "init tensor size not larger than 0";
  // Get metadata
  auto bound = BytePSGlobal::GetPartitionBound();
  auto &name = context.tensor_name;
  context.buff_len = size;
  size_t accumulated = 0;

  // Add for timeline
  BytePSGlobal::SetProfileFlag(&context);
  context.local_rank = BytePSGlobal::GetLocalRank();

  // Total key space is [0, 2^64 - 1]
  // It will be divided to N PS servers, for now we assume N <= 2^16
  // Then we have 2^48 key space left.
  // Top 16 bits out of the 48 bits encodes the sender rank
  // Mid 16 bits out of the 48 bits encodes the tensor id
  // The next 6 bits encodes request types (pushpull, send, etc)
  // The last 10 bits encodes the partition id
  // Therefore, we support up to 2^16 tensors, and up to 2^10 partitions per tensor
  ps::Key start_key = context.declared_key << 16;
  // TODO: support compression in the future
  start_key += (uint32_t) PUSH_PULL_OP << 10;
  while (accumulated < size) {
    context.key_list.push_back(start_key++);
    accumulated +=
        ((size - accumulated) > bound) ? bound : (size - accumulated);
  }

  BPS_LOG(DEBUG) << name << " partitioned to " << context.key_list.size()
                 << " part(s)"
                 << ", total_len=" << size << ", key_range=["
                 << context.key_list.front() << ", " << context.key_list.back()
                 << "]" << " rank=" << BytePSGlobal::GetRank();

  auto key_list = context.key_list;

  BPS_CHECK_GT(key_list.size(), 0) << name;
  BPS_CHECK_EQ(key_list.size(),
               (unsigned int)(size + bound - 1) / bound)  // round up
      << key_list.size() << ", size=" << size << ", bound=" << bound;

  BPS_LOG(TRACE) << "Begin init " << name << ", size=" << size
                 << ", parts=" << key_list.size();

  // If cpubuff is not nullptr, the tensor itself is on CPU
  // We need to register with CUDA so that NCCL can work on it
  if (cpubuff) {
#if BYTEPS_BUILDING_CUDA == 1
    BPS_LOG(DEBUG) << name << " is already on cpu, len=" << size;
    cudaError_t e = cudaHostRegister(cpubuff, size, cudaHostRegisterMapped);
    if (e != cudaSuccess) {
      BPS_LOG(INFO) << cudaGetErrorString(e)
                    << " (You may ignore this if your program continues)";
    }
    CUDA_CALL(cudaHostGetDevicePointer(&(context.gpu_ptr), cpubuff, 0));
#else
    context.gpu_ptr = cpubuff;
#endif
  }

  // We always allocate our own cpu buffer
  // use the first key in key_list as the index
  auto shm_obj = BytePSGlobal::GetSharedMemoryObj();

  size_t aligned_size = Align(size, dtype);
  if (BytePSGlobal::IsCrossPcieSwitch()) {
    // TODO: add BytePS UUID for openPcieSharedMemory and update the corresponding name in RDMAVan
    context.pcie_cpubuff =
        shm_obj->openPcieSharedMemory(std::string("BytePS_Pcie"), key_list[0], aligned_size);
    context.cpubuff = context.pcie_cpubuff.back();
  } else {
    auto shm_prefix = std::string("BytePS_Numa_") + BytePSGlobal::GetUUID() + "_";
    context.numa_cpubuff = shm_obj->openNumaSharedMemory(shm_prefix, key_list[0], aligned_size);
    if (cpubuff) {
      context.cpubuff = context.numa_cpubuff[BytePSGlobal::GetLocalRank()];
    } else {
      context.cpubuff = context.numa_cpubuff[0];
    }
  }
  BPS_LOG(TRACE) << name << ": open shared memory size " << aligned_size;

  // Init tensors with BytePS server
  char *data = const_cast<char *>(static_cast<const char *>(context.cpubuff));
  accumulated = 0;
  size_t i = 0;
  // small tensor does not need to be compressed
  if (size < BytePSGlobal::GetMinCompressBound()) {
    context.kwargs.clear();
  }

  if (BytePSGlobal::IsDistributed() && BytePSGlobal::IsJoint()) {
    // in joint mode every byteps worker instantiates PS.
    auto ps = BytePSGlobal::GetOrInitPS();
  }

  while (accumulated < size) {
    auto key = key_list[i];
    int len = ((size - accumulated) > bound) ? bound : (size - accumulated);
    if (BytePSGlobal::IsDistributed() && BytePSGlobal::IsRootDevice()) {
      auto ps = BytePSGlobal::GetOrInitPS();
      // encode the key for pskv scattering
      auto &pskv = BytePSGlobal::EncodeDefaultKey(key, len);
      // false means not to delete data when SArray is deleted
      ps::SArray<char> vals(data + accumulated, len, false);
      // cmd type
      int cmd = server::GetCommandType(server::RequestType::kLeaderPushPull, dtype, CPU);
      // blocking push, also as a global barrirer
      ps->Wait(ps->ZPush(pskv.keys, vals, pskv.lens, cmd));
      BPS_LOG(TRACE) << "registereed with server, key " << key;

      // register
      if (!context.kwargs.empty()) {
        auto compressor_ptr = compressor::CompressorRegistry::Create(
            context.kwargs, Align(len, dtype), static_cast<DataType>(dtype));
        context.compressor_list.push_back(std::move(compressor_ptr));
      }
    }

    accumulated += len;
    ++i;
  }
  BPS_CHECK_EQ(i, key_list.size());
  BPS_CHECK_EQ(accumulated, size);

  // send to server
  if (!context.kwargs.empty() && BytePSGlobal::IsDistributed() &&
      BytePSGlobal::IsRootDevice()) {
    auto ps = BytePSGlobal::GetOrInitPS();
    auto content = compressor::Serialize(context.kwargs);
    auto len = content.size();
    auto data = const_cast<char *>(content.c_str());
    for (auto key : key_list) {
      auto &kv = BytePSGlobal::EncodeDefaultKey(key, len);
      ps::SArray<char> vals(data, len, false);
      int cmd = server::GetCommandType(server::RequestType::kCompressedPushPull, dtype, CPU);
      ps->Wait(ps->ZPush(kv.keys, vals, kv.lens, cmd));
    }
  }

  context.initialized = true;

  BPS_LOG(TRACE) << "Finish Init " << name << ", size=" << size
                 << ", parts=" << key_list.size();
}

BPSContext &GetContextFromName(const std::string &name) {
  return BytePSGlobal::GetContextFromName(name);
}

bool IsTensorDeclared(const std::string &name) {
  return BytePSGlobal::IsTensorDeclared(name);
}

void RegisterCompressor(const std::string &name,
                        std::unordered_map<std::string, std::string> &kwargs) {
  return BytePSGlobal::RegisterCompressor(name, kwargs);
}

void PinMemory(void* ptr, int numa_node, size_t bytes) {
  return BytePSGlobal::PinMemory(ptr, numa_node, bytes);
}

int32_t IsTensorDeclaredP2P(const std::string &name, int sender, int receiver, int32_t provided_key) {
  return BytePSGlobal::IsTensorDeclaredP2P(name, sender, receiver, provided_key);
}

std::shared_ptr<std::vector<QueueType>> GetSendQueueList() {
  auto queue_list = std::make_shared<std::vector<QueueType>>();
  queue_list->push_back(SEND);
  return queue_list;
}

std::shared_ptr<std::vector<QueueType>> GetRecvQueueList() {
  auto queue_list = std::make_shared<std::vector<QueueType>>();
  // Copy from CPU to GPU
  queue_list->push_back(RECV);
  return queue_list;
}

std::shared_ptr<std::vector<QueueType>> GetPushQueueListGPU(int device) {
  auto queue_list = std::make_shared<std::vector<QueueType>>();

#if BYTEPS_BUILDING_CUDA == 1
  // Per-PCIe-switch NCCL reduce
  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    queue_list->push_back(REDUCE);
  } else {
    queue_list->push_back(COORDINATE_REDUCE);
    queue_list->push_back(REDUCE);
  }
#endif
  // Copy from GPU to CPU
  if (BytePSGlobal::IsDistributed() || BytePSGlobal::IsCrossPcieSwitch()) {
    queue_list->push_back(COPYD2H);
  }

  // Cross-PCIe-switch reduce
  if (BytePSGlobal::IsCrossPcieSwitch()) {
    queue_list->push_back(PCIE_REDUCE);
  }

  // Push in distributed mode
  // In case IsCrossPcieSwitch(), PUSH runs as a dummy barrier
  if (BytePSGlobal::IsDistributed() || BytePSGlobal::IsCrossPcieSwitch()) {
    if (BytePSGlobal::IsRootDevice()) {
      queue_list->push_back(PUSH);
    } else {
      queue_list->push_back(COORDINATE_PUSH);
    }
  }
  return queue_list;
}

std::shared_ptr<std::vector<QueueType>> GetPushQueueListCPU(int device) {
  auto queue_list = std::make_shared<std::vector<QueueType>>();

  if (BytePSGlobal::IsRootDevice()) {
    queue_list->push_back(CPU_COPY);
    queue_list->push_back(CPU_REDUCE);
  } else {
    queue_list->push_back(CPU_COPY);
    queue_list->push_back(CPU_REDUCE);
  }

  // Push in distributed mode
  // In case IsCrossPcieSwitch(), PUSH runs as a dummy barrier
  if (BytePSGlobal::IsDistributed() || BytePSGlobal::IsCrossPcieSwitch()) {
    if (BytePSGlobal::IsRootDevice()) {
      queue_list->push_back(PUSH);
    }
  }
  return queue_list;
}

std::shared_ptr<std::vector<QueueType>> GetPushQueueList(int device) {
  if (device == CPU_DEVICE_ID) {
    return GetPushQueueListCPU(device);
  }
  return GetPushQueueListGPU(device);
}

std::shared_ptr<std::vector<QueueType>> GetPullQueueListGPU(int device) {
  auto queue_list = std::make_shared<std::vector<QueueType>>();

  // Pull in distributed mode
  if (BytePSGlobal::IsDistributed()) {
    if (BytePSGlobal::IsRootDevice()) {
      queue_list->push_back(PULL);
    }
  }

  // Copy from CPU to GPU
  if (BytePSGlobal::IsDistributed() || BytePSGlobal::IsCrossPcieSwitch()) {
    queue_list->push_back(COPYH2D);
  }

#if BYTEPS_BUILDING_CUDA == 1
  // Per-PCIe-switch NCCL broadcast
  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    queue_list->push_back(BROADCAST);
  } else {
    queue_list->push_back(COORDINATE_BROADCAST);
    queue_list->push_back(BROADCAST);
  }
#endif
  return queue_list;
}

std::shared_ptr<std::vector<QueueType>> GetPullQueueListCPU(int device) {
  auto queue_list = std::make_shared<std::vector<QueueType>>();

  // Pull in distributed mode
  if (BytePSGlobal::IsDistributed()) {
    if (BytePSGlobal::IsRootDevice()) {
      queue_list->push_back(PULL);
    }
  }

  queue_list->push_back(CPU_BCAST);
  if (BytePSGlobal::IsRootDevice()) {
    queue_list->push_back(CPU_BCAST_FINISH);
  }
  return queue_list;
}

std::shared_ptr<std::vector<QueueType>> GetPullQueueList(int device) {
  if (device == CPU_DEVICE_ID) {
    return GetPullQueueListCPU(device);
  }
  return GetPullQueueListGPU(device);
}

void print_queue_list(std::shared_ptr<std::vector<QueueType>> queue_list,
                      std::string &name, bool is_dist_reduce_root_node)
{
  BPS_LOG(DEBUG) << "queue_list for tensor: " << name
                 << ", is_dist_reduce_root_node: " << is_dist_reduce_root_node;

  for (auto item : *queue_list) {
    BPS_LOG(DEBUG) << "    " << LogStrings[item];
  }
}

}  // namespace common
}  // namespace byteps
