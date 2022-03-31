// Copyright 2019 Bytedance Inc. All Rights Reserved.
// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#include <torch/extension.h>
#include <torch/torch.h>
#include <chrono>
#include <memory>
#include <thread>

#include "../common/operations.h"
#include "adapter.h"
#include "ops.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "ready_event.h"

namespace byteps {
namespace torch {

static HandleManager handle_manager;

namespace {

std::string GetOpName(const std::string& prefix, const std::string& name,
                      int handle) {
  if (!name.empty()) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

int GetDeviceID(const ::torch::Tensor& tensor) {
  if (tensor.device().is_cuda()) {
    return tensor.device().index();
  }
  return CPU_DEVICE_ID;
}

// zhuang: communication types
enum BYTEPSCommType {
  COMM_PUSH_PULL,
  COMM_PULL_PUSH,
  COMM_CPU_COMPRESS,
  COMM_INTRA_REDUCE,
  COMM_INTRA_GATHER,
  COMM_INTRA_BROADCAST,
  COMM_INTRA_REDUCESCATTER,
  COMM_INTRA_ALLGATHER,
  COMM_INTRA_ALLTOALL,
};

}  // namespace


void StartIntraTask(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string tensor_name, int version, int priority, int handle, BYTEPSCommType comm_type, int root) {
  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto byteps_input = std::make_shared<TorchTensor>(tensor);
  auto byteps_output = std::make_shared<TorchTensor>(output);
  size_t size = byteps_input->size();
  auto dtype = byteps_input->dtype();

  auto& context = common::GetContextFromName(tensor_name);
  std::shared_ptr<std::vector<QueueType>> queue_list=nullptr;

  common::InitIntraTensor(context, size, dtype,
                      (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr,
                      root);
  
  if (comm_type == COMM_INTRA_GATHER) {
    queue_list = common::GetIntraGatherQueueList(device);
  } else if (comm_type == COMM_INTRA_BROADCAST) {
    queue_list = common::GetIntraBroadcastQueueList(device);
  } else if (comm_type == COMM_INTRA_REDUCESCATTER) {
    queue_list = common::GetIntraReducescatterQueueList(device);
  } else if (comm_type == COMM_INTRA_ALLGATHER) {
    queue_list = common::GetIntraAllgatherQueueList(device);
  } else if (comm_type == COMM_INTRA_ALLTOALL) {
    queue_list = common::GetIntraAlltoallQueueList(device);
  } else if (comm_type == COMM_INTRA_REDUCE) {
    queue_list = common::GetIntraReduceQueueList(device);
  }

  auto enqueue_result = common::EnqueueIntraTensor(
    context, byteps_input, byteps_output, ready_event, device, priority,
    version,
    [handle, average, tensor, output](const Status& status) mutable {
      // Will execute in the `device` context.
      if (average) {
#if TORCH_VERSION >= 1005000000
        if (isIntegralType(output.scalar_type())) {
          output.floor_divide_(byteps_local_size());
          handle_manager.MarkDone(handle, status);
          return;
        }
#endif
        output.div_(byteps_local_size());
      }
      handle_manager.MarkDone(handle, status);
    },
    queue_list);
  ThrowIfError(enqueue_result);

  return;
}


void StartTask(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string tensor_name, int version, int priority, int handle, BYTEPSCommType comm_type) {
  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto byteps_input = std::make_shared<TorchTensor>(tensor);
  auto byteps_output = std::make_shared<TorchTensor>(output);
  size_t size = byteps_input->size();
  auto dtype = byteps_input->dtype();

  auto& context = common::GetContextFromName(tensor_name);
  std::shared_ptr<std::vector<QueueType>> queue_list=nullptr;

  if (comm_type == COMM_CPU_COMPRESS) {
    common::InitCPUCompressTensor(context, size, dtype,
                      (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr);
  } else {
    common::InitTensor(context, size, dtype,
                      (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr);
  }
  
  if (comm_type == COMM_PUSH_PULL) {
    queue_list = common::GetPushQueueList(device);
    auto queue_list_pull = common::GetPullQueueList(device);
    queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());
  } else if (comm_type == COMM_CPU_COMPRESS) {
    queue_list = common::GetCPUCompressQueueList(device);
  }

  auto enqueue_result = common::EnqueueTensor(
      context, byteps_input, byteps_output, ready_event, device, priority,
      version,
      [handle, average, tensor, output](const Status& status) mutable {
        // Will execute in the `device` context.
        if (average) {
#if TORCH_VERSION >= 1005000000
          if (isIntegralType(output.scalar_type())) {
            output.floor_divide_(byteps_size());
            handle_manager.MarkDone(handle, status);
            return;
          }
#endif
          output.div_(byteps_size());
        }
        handle_manager.MarkDone(handle, status);
      },
      queue_list);

  ThrowIfError(enqueue_result);
  return;
}


int DoPushPull(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority) {
  ThrowIfError(common::CheckInitialized());  
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartTask(tensor, output, average, tensor_name, version, priority, handle, COMM_PUSH_PULL);
  } else {
    std::thread t(StartTask, tensor, output, average, tensor_name, version, priority, handle, COMM_PUSH_PULL);
    t.detach();
  }
  return handle;
}

int DoCPUCompress(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority) {
  ThrowIfError(common::CheckInitialized());  
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps_cpu_compress", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartTask(tensor, output, average, tensor_name, version, priority, handle, COMM_CPU_COMPRESS);
  } else {
    std::thread t(StartTask, tensor, output, average, tensor_name, version, priority, handle, COMM_CPU_COMPRESS);
    t.detach();
  }
  return handle;
}

void SetNumGrads(int num_grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  num_grads_ = num_grads;
  grad_count_ = 0;
  return;
}


int DoIntraGather(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority, int root) {
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps_intra_gather", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartIntraTask(tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_GATHER, root);
  } else {
    std::thread t(StartIntraTask, tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_GATHER, root);
    t.detach();
  }
  return handle;
}


int DoIntraBroadcast(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority, int root) {
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps_intra_broadcast", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartIntraTask(tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_BROADCAST, root);
  } else {
    std::thread t(StartIntraTask, tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_BROADCAST, root);
    t.detach();
  }
  return handle;
}


int DoIntraReducescatter(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority) {
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps_intra_reducescatter", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartIntraTask(tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_REDUCESCATTER, -1);
  } else {
    std::thread t(StartIntraTask, tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_REDUCESCATTER, -1);
    t.detach();
  }
  return handle;
}


int DoIntraAllgather(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority) {
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps_intra_allgather", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartIntraTask(tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_ALLGATHER, -1);
  } else {
    std::thread t(StartIntraTask, tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_ALLGATHER, -1);
    t.detach();
  }
  return handle;
}


int DoIntraAlltoall(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority) {
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps_intra_alltoall", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartIntraTask(tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_ALLTOALL, -1);
  } else {
    std::thread t(StartIntraTask, tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_ALLTOALL, -1);
    t.detach();
  }
  return handle;
}

int DoIntraReduce(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority, int root) {
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps_intra_reduce", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartIntraTask(tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_REDUCE, root);
  } else {
    std::thread t(StartIntraTask, tensor, output, average, tensor_name, version, priority, handle, COMM_INTRA_REDUCE, root);
    t.detach();
  }
  return handle;
}


int PollHandle(int handle) { return handle_manager.PollHandle(handle) ? 1 : 0; }

void DeclareTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
}


void DeclareCPUCompressTensor(const std::string& name, size_t size) {
  std::string tensor_name = GetOpName("byteps_cpu_compress", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name, size);
}

void DeclareIntraGatherTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps_intra_gather", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
}

void DeclareIntraBroadcastTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps_intra_broadcast", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
}

void DeclareIntraReducescatterTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps_intra_reducescatter", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
}

void DeclareIntraAllgatherTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps_intra_allgather", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
}

void DeclareIntraAlltoallTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps_intra_alltoall", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
}

void DeclareIntraReduceTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps_intra_reduce", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
}

void DeclareInterCompressTensor(const std::string& name, size_t size) {
  std::string tensor_name = GetOpName("byteps_inter_compress", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name, size);
}


void WaitAndClear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

pybind11::tuple DoPushPullGroupSync(::torch::Tensor tensor,
                                    ::torch::Tensor output, int average,
                                    const std::string& name, int version,
                                    int priority) {
  ThrowIfError(common::CheckInitialized());
  std::cout << "do push pull group\n";
  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  int curr_count;

  if (context.initialized) {
    StartTask(tensor, output, average, tensor_name, version, priority, handle, COMM_PUSH_PULL);
  } else {
    std::thread t(StartTask, tensor, output, average, tensor_name, version,
                  priority, handle, COMM_PUSH_PULL);
    t.detach();
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_count_++;
    curr_count = grad_count_;
    if (grad_count_ == num_grads_) {
      grad_count_ = 0;
    }
  }

  return pybind11::make_tuple(handle, curr_count);
}

PYBIND11_MODULE(c_lib, m) {
  // push_pull
  m.def("byteps_torch_push_pull_async_torch_ByteTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_IntTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_LongTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_HalfTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_FloatTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_DoubleTensor", &DoPushPull);

  // cpu_compress
  m.def("byteps_torch_cpu_compress_async_torch_ByteTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_IntTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_LongTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_HalfTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_FloatTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_DoubleTensor", &DoCPUCompress);

  // intra_gather
  m.def("byteps_torch_intra_gather_async_torch_ByteTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_IntTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_LongTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_HalfTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_FloatTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_DoubleTensor", &DoIntraGather); 

  // intra_broadcast
  m.def("byteps_torch_intra_broadcast_async_torch_ByteTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_IntTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_LongTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_HalfTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_FloatTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_DoubleTensor", &DoIntraBroadcast);

  // intra_reducescatter
  m.def("byteps_torch_intra_reducescatter_async_torch_ByteTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_IntTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_LongTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_HalfTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_FloatTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_DoubleTensor", &DoIntraReducescatter);

  // intra_allgather
  m.def("byteps_torch_intra_allgather_async_torch_ByteTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_IntTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_LongTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_HalfTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_FloatTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_DoubleTensor", &DoIntraAllgather); 

  // intra_alltoall
  m.def("byteps_torch_intra_alltoall_async_torch_ByteTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_IntTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_LongTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_HalfTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_FloatTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_DoubleTensor", &DoIntraAlltoall); 

  // intra_reduce
  m.def("byteps_torch_intra_reduce_async_torch_ByteTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_IntTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_LongTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_HalfTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_FloatTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_DoubleTensor", &DoIntraReduce); 

  m.def("byteps_torch_set_num_grads", &SetNumGrads);

  m.def("byteps_torch_push_pull_group_sync_torch_ByteTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_IntTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_LongTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_HalfTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_FloatTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_DoubleTensor", &DoPushPullGroupSync);

#if HAVE_CUDA
  m.def("byteps_torch_push_pull_async_torch_cuda_ByteTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_IntTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_LongTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_HalfTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_FloatTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_DoubleTensor", &DoPushPull);

  // cpu_compress
  m.def("byteps_torch_cpu_compress_async_torch_cuda_ByteTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_cuda_IntTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_cuda_LongTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_cuda_HalfTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_cuda_FloatTensor", &DoCPUCompress);
  m.def("byteps_torch_cpu_compress_async_torch_cuda_DoubleTensor", &DoCPUCompress);

  // intra_gather
  m.def("byteps_torch_intra_gather_async_torch_cuda_ByteTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_cuda_IntTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_cuda_LongTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_cuda_HalfTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_cuda_FloatTensor", &DoIntraGather);
  m.def("byteps_torch_intra_gather_async_torch_cuda_DoubleTensor", &DoIntraGather);

  // intra_broadcast
  m.def("byteps_torch_intra_broadcast_async_torch_cuda_ByteTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_cuda_IntTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_cuda_LongTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_cuda_HalfTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_cuda_FloatTensor", &DoIntraBroadcast);
  m.def("byteps_torch_intra_broadcast_async_torch_cuda_DoubleTensor", &DoIntraBroadcast);

  // intra_reducescatter
  m.def("byteps_torch_intra_reducescatter_async_torch_cuda_ByteTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_cuda_IntTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_cuda_LongTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_cuda_HalfTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_cuda_FloatTensor", &DoIntraReducescatter);
  m.def("byteps_torch_intra_reducescatter_async_torch_cuda_DoubleTensor", &DoIntraReducescatter);

  // intra_allgather
  m.def("byteps_torch_intra_allgather_async_torch_cuda_ByteTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_cuda_IntTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_cuda_LongTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_cuda_HalfTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_cuda_FloatTensor", &DoIntraAllgather);
  m.def("byteps_torch_intra_allgather_async_torch_cuda_DoubleTensor", &DoIntraAllgather);

  // intra_alltoall
  m.def("byteps_torch_intra_alltoall_async_torch_cuda_ByteTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_cuda_IntTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_cuda_LongTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_cuda_HalfTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_cuda_FloatTensor", &DoIntraAlltoall);
  m.def("byteps_torch_intra_alltoall_async_torch_cuda_DoubleTensor", &DoIntraAlltoall);

  // intra_reduce
  m.def("byteps_torch_intra_reduce_async_torch_cuda_ByteTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_cuda_IntTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_cuda_LongTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_cuda_HalfTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_cuda_FloatTensor", &DoIntraReduce);
  m.def("byteps_torch_intra_reduce_async_torch_cuda_DoubleTensor", &DoIntraReduce);

  m.def("byteps_torch_push_pull_group_sync_torch_cuda_ByteTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_IntTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_LongTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_HalfTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_FloatTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_DoubleTensor", &DoPushPullGroupSync);
#endif

  // basics
  m.def("byteps_torch_poll", &PollHandle);
  m.def("byteps_torch_wait_and_clear", &WaitAndClear);
  m.def("byteps_torch_declare_tensor", &DeclareTensor);
  // declare the compressed tensor with the original tensor size
  m.def("byteps_torch_declare_intra_gather_tensor", &DeclareIntraGatherTensor);
  m.def("byteps_torch_declare_intra_broadcast_tensor", &DeclareIntraBroadcastTensor);
  m.def("byteps_torch_declare_intra_reducescatter_tensor", &DeclareIntraReducescatterTensor);
  m.def("byteps_torch_declare_intra_allgather_tensor", &DeclareIntraAllgatherTensor);
  m.def("byteps_torch_declare_intra_alltoall_tensor", &DeclareIntraAlltoallTensor);
  m.def("byteps_torch_declare_intra_reduce_tensor", &DeclareIntraReduceTensor);
  m.def("byteps_torch_declare_inter_compress_tensor", &DeclareInterCompressTensor);
  m.def("byteps_torch_declare_cpu_compress_tensor", &DeclareCPUCompressTensor);
}

}  // namespace torch
}  // namespace byteps
