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
  BPS_LOG(INFO) << "byteps_init() DONE. rank=" << BytePSGlobal::GetRank();
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
  // TODO(haibin.lin): handle mod logic inside byteps_session_id
  auto session_id = common::byteps_session_id(name.c_str());
  int session_size = common::byteps_session_size();
  *session_name = "session_" + std::to_string(session_id % session_size) + "_" + name;
  for (int i = 0; i < tensor_key.size(); ++i) {
    std::string sess_prefix = "session_" + std::to_string(i) + "_";
    common::IsTensorDeclaredAlltoall(sess_prefix + name, tensor_key[i]);
  }
  // Example names used for alltoall:
  // - node_name: my_node
  // - ctx->tensor_name: session_0_my_node
  // - request_task->tensor_name: ssion_0_my_node_request
  // - response_task->tensor_name: session_0_my_node_request_i_resp_j
  auto& bps_context = common::GetContextFromName(*session_name);
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
                             bool output_size_unknown, bool use_pull) {
  if (BytePSGlobal::ShouldShutdown()) {
    return Status::OK();
  }
  // This function prepares the P2PTensorTableEntry for both pull-based and push-based
  // alltoall operations. In general, we construct two types of tasks: request task
  // and response task.
  // For the request task, there will be at most 1 such a task enqueued. For push-based alltoall,
  // it correspond to the "push request", where `my_rank` will push data to all other ranks
  // if there's data (indicated by `send_begin` indices). For pull-based alltoall, it correspond
  // to the "pull request", where `my_rank` will pull data from all other ranks if there's data
  // (indicatd by `recv_begin` indices).
  // For the response task, there will be at most `num_ranks` tasks overall. For push-based
  // alltoall, it refers to the operation that copies from recv buffer to output buffers allocated
  // by frameworks. For pull-based alltoall, it refers to the operation that responses to the
  // requester with actual data.
  BPS_CHECK(!use_pull || !output_size_unknown)
    << "pull-based alltoall does not support unknown recv_spilt";
  // send_begin always starts with a zero
  int num_ranks = send_begin.size() - 1;
  std::shared_ptr<std::atomic_int> counter_ptr(new std::atomic_int(0));
  auto dtype = input ? input->dtype() : group_inputs[0]->dtype();
  auto unit_size = getDataTypeLength(dtype);
  const int my_rank = common::byteps_rank();
  auto& byteps_context = common::GetContextFromName(name);
  bool recv_on_gpu = output_device != CPU_DEVICE_ID;

  // ========= basic task info ==========
  // if use_pull, request_task->offset_list is based on recv_begin
  const std::vector<int>& request_begin = use_pull ? recv_begin : send_begin;
  const std::vector<int>& resp_begin = use_pull ? send_begin : recv_begin;
  std::vector<QueueType> request_q = GetAlltoallRequestQueueList(use_pull);
  std::vector<QueueType> response_q = GetAlltoallResponseQueueList(use_pull, output_size_unknown);

  auto request_task = new P2PTensorTableEntry(priority, version, ready_event, callback,
                                              input_device, request_q, output_device, output_size_unknown,
                                              group_inputs, group_outputs);
  P2PTensorTableEntry base_resp_task(priority, version, ready_event, callback,
                                    input_device, response_q, output_device, output_size_unknown,
                                    group_inputs, group_outputs);
  // the accumulated offset list always starts with 0
  request_task->offset = 0;
  request_task->offset_list.push_back(0);
  request_task->context = &byteps_context;
  request_task->counter_ptr = counter_ptr;
  request_task->tensor = input;
  request_task->output = output;

  base_resp_task.offset_list.push_back(0);
  base_resp_task.context = &byteps_context;
  base_resp_task.counter_ptr = counter_ptr;
  base_resp_task.tensor = input;
  base_resp_task.output = output;

  // ========= Init tensor ==========
  // the number of valid ps-lite send operations
  int num_ps_requests = 0;
  // the number of valid recv tasks (including local memcpy)
  int resp_total_partnum = output_size_unknown ? 1 : 0;
  // prepare size info for initialization / sanity checks
  std::vector<int> request_size_list;
  std::vector<int> resp_size_list;
  for (size_t i = 0; i < num_ranks; ++i) {
    // send to rank i
    int request_size = unit_size * (request_begin[i+1] - request_begin[i]);
    request_size_list.emplace_back(request_size);
    request_task->offset_list.push_back(request_begin[i + 1] * unit_size);
    // we check the case for valid ps-lite send operations
    if (output_size_unknown && i != my_rank) {
      num_ps_requests += 1;
    } else if (i != my_rank && request_size != 0) {
      num_ps_requests += 1;
    }
    // recv from rank i
    int resp_size = unit_size * (resp_begin[i + 1] - resp_begin[i]);
    resp_size_list.emplace_back(resp_size);
    base_resp_task.offset_list.push_back(resp_begin[i + 1] * unit_size);
    // check the case for empty recv tasks
    if (!output_size_unknown && resp_size != 0) {
      resp_total_partnum += 1;
    }
  }
  // calculate the number of partitions
  int request_total_partnum = 1;
  if (!output_size_unknown && num_ps_requests == 0) {
    request_total_partnum = 0;
  }
  int total_partnum = request_total_partnum + resp_total_partnum;
  request_task->total_partnum = total_partnum;
  base_resp_task.total_partnum = total_partnum;

  // initialize the key list and buffer list
  common::InitTensorAlltoall(byteps_context, request_size_list, resp_size_list,
                             dtype, recv_on_gpu, use_pull);
  BPS_CHECK(byteps_context.cpubuff_list.size() == num_ranks * 2);
  BPS_CHECK(byteps_context.key_list.size() == num_ranks * 2);
  // the first half of cpubuff_list/key_list is for the request task
  request_task->pcie_cpubuff.insert(request_task->pcie_cpubuff.begin(),
                                    byteps_context.cpubuff_list.begin(),
                                    byteps_context.cpubuff_list.begin() + num_ranks);
  request_task->key_list.insert(request_task->key_list.begin(),
                             byteps_context.key_list.begin(),
                             byteps_context.key_list.begin() + num_ranks);
  // the second half of cpubuff_list/key_list is for the response task
  base_resp_task.pcie_cpubuff.insert(base_resp_task.pcie_cpubuff.begin(),
                                    byteps_context.cpubuff_list.begin() + num_ranks,
                                    byteps_context.cpubuff_list.end());
  base_resp_task.key_list.insert(base_resp_task.key_list.begin(),
                                byteps_context.key_list.begin() + num_ranks,
                                byteps_context.key_list.end());
  BPS_CHECK(request_task->pcie_cpubuff.size() == num_ranks);
  BPS_CHECK(request_task->key_list.size() == num_ranks);
  BPS_CHECK(base_resp_task.pcie_cpubuff.size() == num_ranks);
  BPS_CHECK(base_resp_task.key_list.size() == num_ranks);

  request_task->request_counter = std::make_shared<std::atomic_int>(num_ps_requests);

  // enqueue send tasks
  if (request_total_partnum) {
    request_task->offset = 0; // DO WE NEED THIS? use default value?
    request_task->tensor_name = name + "_request";
    BytePSGlobal::GetScheduledQueue(request_q.at(0))->addTask(request_task);
  } else {
    // nothing to send.
    delete request_task;
  }

  // enqueue recv tasks
  if (total_partnum == 0) {
    callback(Status::OK());
  } else if (resp_total_partnum) {
    base_resp_task.offset = 0;
    base_resp_task.aux_output = size_output;
    if (output_size_unknown) {
      base_resp_task.tensor_name = name + "_resp";
      base_resp_task.key = server::GetAlltoallTensorId(base_resp_task.key_list.at(0));
      auto resp_task = new P2PTensorTableEntry(base_resp_task);
      BytePSGlobal::GetScheduledQueue(response_q.at(0))->addTask(resp_task);
    } else {
      // naming for sub-tasks
      std::string recv_name_suffix = "_resp_" + std::to_string(my_rank);
      std::string send_name_prefix = name + "_request_";
      for (size_t i = 0; i < num_ranks; ++i) {
        int resp_size = unit_size * (resp_begin[i + 1] - resp_begin[i]);
        if (resp_size) {
          auto resp_task = new P2PTensorTableEntry(base_resp_task);
          resp_task->tensor_name = send_name_prefix + std::to_string(i) + recv_name_suffix;;
          resp_task->key = base_resp_task.key_list.at(i);
          resp_task->len = resp_size;
          resp_task->offset = group_inputs.size() ? 0 : resp_task->offset_list[i];
          if (i == my_rank) {
            // for local send-recv, we need to know the offset of the input tensor
            resp_task->offset = request_begin[i] * unit_size;
            BytePSGlobal::GetScheduledQueue(response_q.at(0))->addTask(resp_task);
            auto table = use_pull ? BytePSGlobal::GetP2PPullResponseTable() : BytePSGlobal::GetP2PCopyTable();
            table->AddReadyCount(resp_task->key);
          } else {
            BytePSGlobal::GetScheduledQueue(response_q.at(0))->addTask(resp_task);
          }
        }
      }
    }
  }
  BPS_LOG(TRACE) << "EnqueueAlltoAllTensor finished: " << name
                << " rank=" << BytePSGlobal::GetRank()
                << " request_partnum=" << request_total_partnum
                << " resp_partnum=" << resp_total_partnum
                << " num_ps_requests=" << num_ps_requests;
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

void GenerateAlltoallKeys(std::vector<uint64_t>* key_list, int32_t declared_key,
                          size_t num_ranks) {
  auto my_rank = BytePSGlobal::GetRank();
  // Generate alltoall key_list for all request tasks (keys are the same)
  ps::Key request_key = server::ComposeAlltoallKey(declared_key, my_rank);
  for (size_t i = 0; i < num_ranks; ++i) {
    key_list->push_back(request_key);
  }
  // Generate alltoall key_list for all response tasks
  for (size_t i = 0; i < num_ranks; ++i) {
    ps::Key resp_key = server::ComposeAlltoallKey(declared_key, i);
    key_list->push_back(resp_key);
  }
}

void InitTensorAlltoall(BPSContext &context, std::vector<int> &request_size_list,
                        std::vector<int> &resp_size_list, int dtype,
                        bool recv_on_gpu, bool use_pull) {
  auto bound = BytePSGlobal::GetAlltoallBuffBound();
  std::lock_guard<std::mutex> lock(context.init_mutex);
  size_t total_request_size = 0;
  size_t total_resp_size = 0;
  // TODO(haibin.lin): we only support 1 partition per send/recv pair for alltoall
  for (size_t i = 0; i < request_size_list.size(); ++i) {
    BPS_CHECK(request_size_list[i] <= bound) << "Alltoall send size exceeds buffer size for rank="
      << i << " name=" << context.tensor_name << " size=" << request_size_list[i];
    BPS_CHECK(resp_size_list[i] <= bound) << "Alltoall recv size exceeds buffer size for rank="
      << i << " name=" << context.tensor_name << " size=" << resp_size_list[i];
    total_request_size += request_size_list[i];
    total_resp_size += resp_size_list[i];
  }
  if (context.initialized) {
    return;
  }
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));

  // Add for timeline
  BytePSGlobal::SetProfileFlag(&context);
  context.local_rank = BytePSGlobal::GetLocalRank();

  // Generate alltoall key_list for recv tasks
  size_t num_ranks = request_size_list.size();
  GenerateAlltoallKeys(&context.key_list, context.declared_key, num_ranks);
  BPS_LOG(DEBUG) << "InitTensorAlltoall: " << context.tensor_name << " request_size=" << total_request_size
                 << " resp_size=" << total_resp_size << " rank=" << BytePSGlobal::GetRank();
  auto& key_list = context.key_list;
  BPS_CHECK(num_ranks * 2 == key_list.size()) << key_list.size();

  // P2P operations do not need to register tensor with CUDA for NCCL
  context.gpu_ptr = nullptr;

  // We always allocate our own cpu buffer
  // use the first key in key_list as the index
  auto shm_obj = BytePSGlobal::GetSharedMemoryObj();
  // use a different prefix for p2p tensors
  std::string wid = "_" + std::to_string(BytePSGlobal::GetWorkerID()) + "_";
  std::string shm_name = std::string("BytePS_P2P_ShM_") + BytePSGlobal::GetUUID() + wid;
  context.cpubuff = nullptr;
  auto my_rank = BytePSGlobal::GetRank();
  CHECK(BytePSGlobal::IsDistributed());
  auto ps = BytePSGlobal::GetOrInitPS();
  // send buffs
  for (size_t i = 0; i < num_ranks; ++i) {
    if (use_pull) {
      context.cpubuff_list.emplace_back(nullptr);
    } else {
      auto k = key_list[i];
      auto sender = my_rank;
      auto receiver = i;
      auto pskv = BytePSGlobal::EncodeP2PKey(k, bound, receiver);
      BPS_LOG(DEBUG) << "Init ps-lite key:" << k << " encoded:" << pskv.keys[0];
      // the shared memory is always created at partition size
      void* buff = shm_obj->openSharedMemory(shm_name, pskv.keys[0], bound);
      context.cpubuff_list.emplace_back(buff);
      // false means not to delete data when SArray is deleted
      ps::SArray<char> vals((char*) buff, bound, false);
      DeviceType device = recv_on_gpu ? GPU : CPU;
      int cmd = server::GetCommandType(server::RequestType::kDefaultSend, dtype, device);
      if (!BytePSGlobal::IsAlltoallUsePull() && sender != receiver) {
        // blocking push, also as a global barrirer
        ps->Wait(ps->ZPush(pskv.keys, vals, pskv.lens, cmd));
      }
    }
  }
  // recv buffs
  for (size_t i = 0; i < num_ranks; ++i) {
    // no need to create the cpubuff as a receiver
    context.cpubuff_list.emplace_back(nullptr);
  }

  context.initialized = true;
  BPS_LOG(TRACE) << "Finish Init " << context.tensor_name << " request_size=" << total_request_size
                 << " resp_size=" << total_resp_size << " use_pull=" << use_pull;
}

void InitTensorP2P(BPSContext &context, size_t size, int dtype, void *cpubuff,
                   int sender, int receiver, bool recv_on_gpu) {
  auto bound = BytePSGlobal::GetPartitionBound();
  std::lock_guard<std::mutex> lock(context.init_mutex);
  BPS_CHECK(size > 0);
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
                 << " part(s), total_len=" << size << ", key_range=["
                 << context.key_list.front() << ", " << context.key_list.back()
                 << "] worker_id=" << BytePSGlobal::GetWorkerID()
                 << ", sender=" << sender << ", receiver=" << receiver;

  auto key_list = context.key_list;
  // P2P operations do not need to register tensor with CUDA for NCCL
  context.gpu_ptr = nullptr;

  // We always allocate our own cpu buffer
  // use the first key in key_list as the index
  auto shm_obj = BytePSGlobal::GetSharedMemoryObj();
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
    // TODO(haibin.lin): We assume the number of partitions do not change
    // When encoding for the first time, declare len = bound
    auto pskv = BytePSGlobal::EncodeP2PKey(k, bound, receiver);
    // the shared memory is always created at partition size
    void* buff = nullptr;
    if (sender == my_rank && sender != receiver) {
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
    BytePSGlobal::GetOrInitPS();
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

int32_t IsTensorDeclared(const std::string &name) {
  return BytePSGlobal::IsTensorDeclared(name, PUSH_PULL_OP, -1);
}

int32_t IsTensorDeclaredAlltoall(const std::string &name, int32_t provided_key) {
  return BytePSGlobal::IsTensorDeclared(name, ALLTOALL_OP, provided_key);
}

void RegisterCompressor(const std::string &name,
                        std::unordered_map<std::string, std::string> &kwargs) {
  return BytePSGlobal::RegisterCompressor(name, kwargs);
}

void PinMemory(void* ptr, int numa_node, size_t bytes) {
  return BytePSGlobal::PinMemory(ptr, numa_node, bytes);
}

int32_t IsTensorDeclaredP2P(const std::string &name, int sender, int receiver) {
  return BytePSGlobal::IsTensorDeclaredP2P(name, sender, receiver);
}

std::shared_ptr<std::vector<QueueType>> GetSendQueueList() {
  auto queue_list = std::make_shared<std::vector<QueueType>>();
  queue_list->push_back(SEND);
  return queue_list;
}

std::shared_ptr<std::vector<QueueType>> GetRecvQueueList() {
  auto queue_list = std::make_shared<std::vector<QueueType>>();
  queue_list->push_back(RECV);
  return queue_list;
}

std::vector<QueueType> GetAlltoallRequestQueueList(bool use_pull) {
  std::vector<QueueType> queue_list;
  queue_list.emplace_back(use_pull ? P2P_PULL : SEND);
  return queue_list;
}

std::vector<QueueType> GetAlltoallResponseQueueList(bool use_pull, bool output_size_unknown) {
  if (use_pull) {
    if (BytePSGlobal::IsP2PAckDisabled()) {
      return {P2P_PULL_RESPONSE};
    } else {
      return {P2P_PULL_RESPONSE, P2P_WAIT_ACK};
    }
  } else {
    if (output_size_unknown) {
      return {P2P_GROUP_COPYH2D};
    } else {
      return {RECV};
    }
  }
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
