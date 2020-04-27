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

#if HAVE_CUDA
#include <cassert>
#include <mutex>
#include <queue>
#include <unordered_map>
#endif

#include "cuda_util.h"
#include "ready_event.h"

namespace byteps {
namespace sparse {

#if HAVE_CUDA
struct ReadyEventRegistry {
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;

GeneralReadyEvent::GeneralReadyEvent(int device, cudaStream_t stream) : device_(device), stream_(stream) {
  assert(device_ != CPU_DEVICE_ID);

  with_device device_context(device_);
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    if (!queue.empty()) {
      cuda_event_ = queue.front();
      queue.pop();
    } else {
      CUDA_CALL(cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
    }
  }
  CUDA_CALL(cudaEventRecord(cuda_event_, stream_));
}

GeneralReadyEvent::~GeneralReadyEvent() {
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    queue.push(cuda_event_);
  }
}

bool GeneralReadyEvent::Ready() const {
  auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  CUDA_CALL(status);
  return true;
}
#endif

// On GPU this event will signal that GPU computations are done and data is
// ready.
std::shared_ptr<ReadyEvent> RecordReadyEvent(int device, cudaStream_t stream) {
  if (device == CPU_DEVICE_ID) {
    return std::shared_ptr<ReadyEvent>();
  } else {
#if HAVE_CUDA
    return std::make_shared<GeneralReadyEvent>(device, stream);
#else
    throw std::logic_error(
        "Internal error. Requested ReadyEvent "
        "with GPU device but not compiled with CUDA.");
#endif
  }
}

}  // namespace sparse
}  // namespace byteps
