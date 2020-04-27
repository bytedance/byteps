// Copyright 2020 Bytedance Inc. All Rights Reserved.
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

#include <chrono>
#include <memory>
#include <thread>

#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "ready_event.h"

namespace byteps {
namespace sparse {

static HandleManager handle_manager;

namespace {

std::string GetOpName(const std::string& prefix, const std::string& name,
                      int handle) {
  if (!name.empty()) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

}  // namespace

void StartTask(void* tensor, void* output, size_t count, ncclDataType_t datatype, cudaStream_t stream,
               int device, const std::string tensor_name, int version, int priority, int handle) {

  auto ready_event = RecordReadyEvent(device, stream);
  auto byteps_input = std::make_shared<GeneralTensor>(tensor, datatype, count);
  auto byteps_output = std::make_shared<GeneralTensor>(output, datatype, count);
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
      [handle, tensor, output](const Status& status) mutable {
        handle_manager.MarkDone(handle, status);
      },
      queue_list);

  ThrowIfError(enqueue_result);
  return;

}

int DoPushPull(void* tensor, void* output, size_t count, ncclDataType_t datatype, cudaStream_t stream, int device,
               const std::string& name, int version, int priority) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized) {
    StartTask(tensor, output, count, datatype, stream, device, tensor_name, version, priority, handle);
  } else {
    std::thread t(StartTask, tensor, output, count, datatype, stream, device, tensor_name, version, priority, handle);
    t.detach();
  }
  return handle;
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

PYBIND11_MODULE(c_lib, m) {
  // push_pull
  m.def("byteps_torch_push_pull_async_torch_IntTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_LongTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_HalfTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_FloatTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_DoubleTensor", &DoPushPull);

#if HAVE_CUDA
  m.def("byteps_torch_push_pull_async_torch_cuda_IntTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_LongTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_HalfTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_FloatTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_DoubleTensor", &DoPushPull);
#endif

  // basics
  m.def("byteps_torch_poll", &PollHandle);
  m.def("byteps_torch_wait_and_clear", &WaitAndClear);
  m.def("byteps_torch_declare_tensor", &DeclareTensor);
}

}  // namespace sparse
}  // namespace byteps
