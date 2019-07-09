// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

#include "ops.h"
#include <atomic>
#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "ready_event.h"
#include "tensor_util.h"

namespace byteps {
namespace mxnet {

namespace {

std::atomic_int op_count;

std::string GetOpName(std::string prefix, char* name) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }

  op_count.fetch_add(1);
  return prefix + ".noname." + std::to_string(op_count);
}
}  // namespace

inline void InvokeCompleteCallback(Callback on_complete, const Status& status) {
  if (status.ok()) {
    on_complete();
  } else {
    auto error = dmlc::Error(status.reason());
    on_complete(&error);
  }
}

void DoPushPull(BPSContext& context, NDArray* input, int version, int priority,
                Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice(input);
  auto byteps_input = std::make_shared<MXTensor<NDArray>>(input);
  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  auto enqueue_result = common::EnqueueTensor(
      context, byteps_input, byteps_input, nullptr, device, priority, version,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      },
      queue_list);
  ThrowIfError(enqueue_result);
}

extern "C" int byteps_mxnet_push_pull_async(NDArray* tensor, char* name,
                                            int version, int priority,
                                            bool is_average) {
  MX_API_BEGIN();

  std::string tensor_name = GetOpName("byteps", name);

  auto& context = common::GetContextFromName(tensor_name);
  auto dtype = TensorUtil::GetDType(tensor);
  auto size = TensorUtil::GetSize(tensor);
  auto device = TensorUtil::GetDevice(tensor);
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(
                            std::make_shared<MXTensor<NDArray>>(tensor)->data())
                      : nullptr;
  common::InitTensor(context, size, dtype, cpubuff);

  auto push_pull_async_fn = [&context, tensor, version, priority](
                                RunContext rctx, Callback on_complete) mutable {
    DoPushPull(context, tensor, version, priority, on_complete);
  };

  Engine::Get()->PushAsync(push_pull_async_fn, Context::CPU(), {},
                           {tensor->var()}, FnProperty::kCPUPrioritized, 0,
                           "BytePSPushPull");

  if (is_average) {
    // average the aggregated gradient
    auto num_worker = byteps_size();
    *tensor /= num_worker;
  }

  MX_API_END();
}

extern "C" void byteps_mxnet_declare_tensor(NDArray* tensor, char* name) {
  std::string tensor_name = GetOpName("byteps", name);
  common::IsTensorDeclared(tensor_name);
  return;
}

}  // namespace mxnet
}  // namespace byteps
