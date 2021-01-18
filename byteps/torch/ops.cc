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

}  // namespace

void StartTask(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string tensor_name, int version, int priority, int handle) {

  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto byteps_input = std::make_shared<TorchTensor>(tensor);
  auto byteps_output = std::make_shared<TorchTensor>(output);
  size_t size = byteps_input->size();
  auto dtype = byteps_input->dtype();

  auto& context = common::GetContextFromName(tensor_name);
  common::InitTensor(context, size, dtype,
                      (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

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
    StartTask(tensor, output, average, tensor_name, version, priority, handle);
  } else {
    std::thread t(StartTask, tensor, output, average, tensor_name, version, priority, handle);
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

int PollHandle(int handle) { return handle_manager.PollHandle(handle) ? 1 : 0; }

void DeclareTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
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

  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  int curr_count;

  if (context.initialized) {
    StartTask(tensor, output, average, tensor_name, version, priority, handle);
  } else {
    std::thread t(StartTask, tensor, output, average, tensor_name, version,
                  priority, handle);
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
}

}  // namespace torch
}  // namespace byteps
