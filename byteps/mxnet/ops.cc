// Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
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

#include <atomic>

#include "../common/operations.h"

#include "adapter.h"
#include "cuda_util.h"
#include "ops.h"
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
} // namespace

inline void InvokeCompleteCallback(Callback on_complete, const Status& status) {
    if (status.ok()) {
      on_complete();
    } else {
      auto error = dmlc::Error(status.reason());
      on_complete(&error);
    }
}

void DoFirstStage(BPSContext &context, NDArray* input, const std::string& name, int version, int priority,
                 Callback on_complete) {
    ThrowIfError(common::CheckInitialized());

    auto device = TensorUtil::GetDevice(input);
    auto byteps_input = std::make_shared<MXTensor<NDArray>>(input);

    std::vector<QueueType> queue_list;

    // TODO: only test case now
    if (common::IsRoot()) {
        //queue_list.push_back(common::REDUCE);
        queue_list.push_back(common::COPYD2H);
        queue_list.push_back(common::PUSH);
    } else {
        //queue_list.push_back(common::COORDINATE);
        //queue_list.push_back(common::REDUCE);
    }

    auto enqueue_result =
        common::EnqueueTensor(context, byteps_input, nullptr, nullptr,
                              name, device, priority, version,
                              [on_complete](const Status& status) {
                                InvokeCompleteCallback(on_complete, status);
                              }, queue_list);
    ThrowIfError(enqueue_result);
}

void DoSecondStage(BPSContext &context, NDArray* output, const std::string& name, int version, int priority,
                 Callback on_complete) {
    ThrowIfError(common::CheckInitialized());

    auto device = TensorUtil::GetDevice(output);
    auto byteps_output = std::make_shared<MXTensor<NDArray>>(output);

    std::vector<QueueType> queue_list;

    // TODO: only test case now
    if (common::IsRoot()) {
        queue_list.push_back(common::PULL);
        queue_list.push_back(common::COPYH2D);
        //queue_list.push_back(common::BROADCAST);
    } else {
        //queue_list.push_back(common::BROADCAST);
    }

    auto enqueue_result =
        common::EnqueueTensor(context, nullptr, byteps_output, nullptr,
                              name, device, priority, version,
                              [on_complete](const Status& status) {
                                InvokeCompleteCallback(on_complete, status);
                              }, queue_list);
    ThrowIfError(enqueue_result);
}

extern "C" int byteps_mxnet_push_pull_async(NDArray* tensor,
                                            char* name, int version, int priority) {
    MX_API_BEGIN();

    // TODO: replace "byteps" with job ID
    std::string tensor_name = GetOpName("byteps", name);

    size_t size = TensorUtil::GetSize(tensor);
    auto dtype = TensorUtil::GetDType(tensor);

    // check if we need to init the tensor
    if (!common::IsTensorInitialized(tensor_name, size)) {
        // we need to init this tensor with PS
        auto& context = common::GetContextFromName(tensor_name);
        auto device = TensorUtil::GetDevice(tensor);
        auto byteps_input = std::make_shared<MXTensor<NDArray>>(tensor);

        // the following init is blocking, in order to guarantee the order
        common::InitTensor(context, tensor_name, dtype,
            (device == CPU_DEVICE_ID) ? const_cast<void*>(byteps_input->data()) : nullptr);
    }

    auto& context = common::GetContextFromName(tensor_name);
    auto first_stage_async_fn = [&context, tensor, tensor_name, version, priority](RunContext rctx,
                                      Callback on_complete) mutable {
        DoFirstStage(context, tensor, tensor_name, version, priority, on_complete);
    };

    Engine::Get()->PushAsync(first_stage_async_fn, tensor->ctx(),
                            {tensor->var()}, {},
                            FnProperty::kNormal, 0, "BytePSFirstStage");

    auto second_stage_async_fn = [&context, tensor, tensor_name, version, priority](RunContext rctx,
                                      Callback on_complete) mutable {
        DoSecondStage(context, tensor, tensor_name, version, priority, on_complete);
    };

    Engine::Get()->PushAsync(second_stage_async_fn, tensor->ctx(),
                            {}, {tensor->var()},
                            FnProperty::kNormal, 0, "BytePSSecondStage");

    // average the aggregated gradient
    auto num_worker = ps::NumWorkers();
    *tensor /= num_worker;

    MX_API_END();
}

} // namespace mxnet
} // namespace byteps
