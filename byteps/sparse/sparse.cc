// Copyright 2020 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include "sparse.h"
#include "../common/operations.h"

namespace byteps {
namespace sparse {

void BytepsReduceScatter(const void* sendbuff, void* recvbuff, size_t count, char* name) {
  std::string tensor_name = std:string(name);
  auto& context = common::GetContextFromName(tensor_name);

  auto byteps_input = sendbuff;
  auto byteps_output = recvbuff;

  common::InitTensor(context, count,
                      (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr);

  auto queue_list = common::GetSparseQueueList(device);
  std::shared_ptr<ReadyEvent> ready_event; // empty placeholder

  auto enqueue_result = 
      common::EnqueueTensor(
          context, 
          byteps_input, 
          byteps_output,
          ready_event,
          device, 
          0, // priority 
          0, // version
          [](const Status& status) mutable {}, // empty callback for now
          queue_list);
  
}

void BytepsAllGather(const void* sendbuff, void* recvbuff, size_t count) {

}

} // namespace sparse
} // namespace byteps 