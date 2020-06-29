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

#include <cuda_runtime.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <thread>

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
}

void byteps_lazy_init() {
  BytePSGlobal::Init();

  // The order of func does not matter
  std::vector<LoopFunction> func;

  // Push & Pull in distributed mode
  if (BytePSGlobal::IsDistributed()) {
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
      func.push_back(NonRootCopyListenLoop);
    }
  }

  // Per-PCIe-switch NCCL calls
  func.push_back(SyncNcclLoop);
  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    func.push_back(RootNcclLoop);
  } else {
    func.push_back(CoordinateReduceLoop);
    func.push_back(CoordinateBroadcastLoop);
    func.push_back(NonRootNcclLoop);
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

}  // extern "C"

Status CheckInitialized() { return BytePSGlobal::CheckInit(); }

void PartitionTensor(
    std::shared_ptr<TensorTableEntry> entry,
    std::vector<std::shared_ptr<TensorTableEntry>> &partitions) {
  BPS_CHECK(entry->counter_ptr)
      << entry->tensor_name << " counter pointer is null";
  auto size = entry->tensor ? entry->tensor->size() : entry->output->size();
  auto bound = BytePSGlobal::GetPartitionBound();
  auto accumulated = 0;
  int i = 0;

  while (accumulated < size) {
    std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
    // will assign the key later, so don't do it now
    // e->key = entry->key;
    e->tensor_name = entry->tensor_name + std::string("_") + std::to_string(i);
    e->context = entry->context;
    e->ready_event = entry->ready_event;
    e->device = entry->device;
    e->priority = entry->priority;
    e->version = entry->version;
    e->callback = entry->callback;
    e->cpubuff = entry->cpubuff;
    e->gpu_ptr = entry->gpu_ptr;
    e->pcie_cpubuff = entry->pcie_cpubuff;
    e->queue_list = entry->queue_list;
    e->tensor = entry->tensor;
    e->output = entry->output;
    e->offset = accumulated;
    e->len = ((size - accumulated) > bound) ? bound : (size - accumulated);
    e->counter_ptr = entry->counter_ptr;
    e->total_partnum = entry->total_partnum;
    if (!entry->context->compressor_list.empty()) {
      e->compressor = entry->context->compressor_list[i];
    }

    accumulated += e->len;
    ++i;

    partitions.push_back(e);
  }
}

Status EnqueueTensor(BPSContext &context, std::shared_ptr<Tensor> input,
                     std::shared_ptr<Tensor> output,
                     std::shared_ptr<ReadyEvent> ready_event, const int device,
                     const int priority, const int version,
                     StatusCallback callback,
                     std::shared_ptr<std::vector<QueueType>> queue_list) {
  if (BytePSGlobal::ShouldShutdown()) {
    return Status::OK();
  }

  auto &name = context.tensor_name;
  if (input && output) {
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

  std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
  e->tensor_name = name;
  e->context = &context;
  e->tensor = input;
  e->output = output;
  e->ready_event = ready_event;
  e->device = device;
  e->priority = priority;
  e->version = version;
  e->callback = callback;

  if (device == CPU_DEVICE_ID) {
    cudaError_t err = cudaHostRegister(const_cast<void *>(input->data()),
                                       input->size(), cudaHostRegisterMapped);
    if (err == cudaSuccess) {
      BPS_LOG(DEBUG) << name
                     << " cpu address has changed, so it is pinned again.";
    }
    CUDA_CALL(cudaHostGetDevicePointer(&(context.gpu_ptr),
                                       const_cast<void *>(input->data()), 0));
  }

  e->cpubuff = context.cpubuff;
  e->gpu_ptr = context.gpu_ptr;
  e->pcie_cpubuff = context.pcie_cpubuff;
  e->queue_list = *queue_list;
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
                   << " rank=" << BytePSGlobal::GetLocalRank();

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

void InitTensor(BPSContext &context, size_t size, int dtype, void *cpubuff) {
  std::lock_guard<std::mutex> lock(context.init_mutex);
  if (context.initialized) {
    return;
  }
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));

  BPS_CHECK_GT(size, 0) << "init tensor size not larger than 0";
  // Get metadata
  auto bound = BytePSGlobal::GetPartitionBound();
  auto &name = context.tensor_name;
  context.buff_len = size;
  size_t accumulated = 0;

  // Add for timeline
  BytePSGlobal::SetProfileFlag(&context);
  context.local_rank = BytePSGlobal::GetLocalRank();

  // Total key space is 0 to 2^64 - 1
  // It will be divided to N PS servers, for now we assume N <= 2^16
  // Then we have 2^48 key space left (top 16 bits for different servers)
  // MXNet server has a bug dealing with keys larger than 2^32
  // Below we support up to 2^16 tensors, and up to 2^16 partitions per tensor
  ps::Key start_key = context.declared_key << 16;
  while (accumulated < size) {
    context.key_list.push_back(start_key++);
    accumulated +=
        ((size - accumulated) > bound) ? bound : (size - accumulated);
  }
  BPS_LOG(DEBUG) << name << " partitioned to " << context.key_list.size()
                 << " part(s)"
                 << ", total_len=" << size << ", key_range=["
                 << context.key_list.front() << ", " << context.key_list.back()
                 << "]"
                 << " rank=" << BytePSGlobal::GetLocalRank();

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
    BPS_LOG(DEBUG) << name << " is already on cpu, len=" << size;
    cudaError_t e = cudaHostRegister(cpubuff, size, cudaHostRegisterMapped);
    if (e != cudaSuccess) {
      BPS_LOG(INFO) << cudaGetErrorString(e)
                    << " (You may ignore this if your program continues)";
    }
    CUDA_CALL(cudaHostGetDevicePointer(&(context.gpu_ptr), cpubuff, 0));
  }

  // We always allocate our own cpu buffer
  // use the first key in key_list as the index
  auto shm_obj = BytePSGlobal::GetSharedMemoryObj();

  size_t aligned_size = Align(size, dtype);
  if (BytePSGlobal::IsCrossPcieSwitch()) {
    context.pcie_cpubuff =
        shm_obj->openPcieSharedMemory(key_list[0], aligned_size);
    context.cpubuff = context.pcie_cpubuff.back();
  } else {
    context.cpubuff = shm_obj->openSharedMemory(std::string("BytePS_ShM_"),
                                                key_list[0], aligned_size);
  }
  BPS_LOG(TRACE) << name << ": open shared memory size " << aligned_size;

  // Init tensors with BytePS server
  char *data = const_cast<char *>(static_cast<const char *>(context.cpubuff));
  accumulated = 0;
  size_t i = 0;
  BPS_LOG(INFO) << "tensor size=" << size;
  // small tensor does not need to be compressed
  if (size < BytePSGlobal::GetMinCompressBound()) {
    context.kwargs.clear();
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
      int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
      // blocking push, also as a global barrirer
      ps->Wait(ps->ZPush(pskv.keys, vals, pskv.lens, cmd));

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

  BPS_CHECK_EQ(accumulated, size);
  BPS_CHECK_EQ(i, key_list.size());

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
      int cmd = GetCommandType(RequestType::kCompressedPushPull, dtype);
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

std::shared_ptr<std::vector<QueueType>> GetPushQueueList(int device) {
  auto queue_list = std::make_shared<std::vector<QueueType>>();

  // Per-PCIe-switch NCCL reduce
  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    queue_list->push_back(REDUCE);
  } else {
    queue_list->push_back(COORDINATE_REDUCE);
    queue_list->push_back(REDUCE);
  }

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

std::shared_ptr<std::vector<QueueType>> GetPullQueueList(int device) {
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

  // Per-PCIe-switch NCCL broadcast
  if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
    queue_list->push_back(BROADCAST);
  } else {
    queue_list->push_back(COORDINATE_BROADCAST);
    queue_list->push_back(BROADCAST);
  }
  return queue_list;
}

}  // namespace common
}  // namespace byteps
