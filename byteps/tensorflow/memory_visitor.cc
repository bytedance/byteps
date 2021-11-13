// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
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

#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>
#include <atomic>
#include <numa.h>

#include "memory_visitor.h"
#include "../common/logging.h"

#include "tensorflow/core/public/version.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/process_state.h"

// the memory visitor header is only available since TF2
#if TF_MAJOR_VERSION >= 2 
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif

using namespace byteps;

namespace byteps {
namespace tensorflow {

class MemoryVistor {
 public:
  MemoryVistor() {
#if TF_MAJOR_VERSION >= 2
#if BYTEPS_BUILDING_CUDA == 1
    auto add_visitor = getenv("BYTEPS_PIN_MEMORY");
    ::tensorflow::SubAllocator::Visitor gpu_alloc_visitor = [](void* ptr, int gpu_id,
                                             size_t num_bytes) {
      common::PinMemory(ptr, gpu_id, num_bytes, true);
    };
    if (add_visitor && atoi(add_visitor)) {
      // XXX we assume only 1 GPU is visible to the current process
      // TODO: TfGpuId is not defined for the latest TF
      auto numa_id = getenv("BYTEPS_NUMA_ID");
      int bus_id = numa_id ? atoi(numa_id) : 0;
      ::tensorflow::GPUProcessState::singleton()->AddGPUAllocVisitor(bus_id, gpu_alloc_visitor);
      BPS_LOG(DEBUG) << "BytePS pinned memory visitor for GPU enabled. numa_id=" << bus_id;
    } else {
      BPS_LOG(DEBUG) << "BytePS pinned memory visitor for GPU NOT enabled";
    }
#endif
#endif
    auto add_visitor_cpu = getenv("BYTEPS_PIN_MEMORY_CPU");
    if (add_visitor_cpu && atoi(add_visitor_cpu)) {
      ::tensorflow::SubAllocator::Visitor alloc_visitor = [](void* ptr, int numa_node,
                                              size_t num_bytes) {
        common::PinMemory(ptr, numa_node, num_bytes, false);
      };
      ::tensorflow::ProcessState::singleton()->AddCPUAllocVisitor(alloc_visitor);
      BPS_LOG(DEBUG) << "BytePS pinned memory visitor for CPU enabled.";
    } else {
      BPS_LOG(DEBUG) << "BytePS pinned memory visitor for CPU NOT enabled";
    }
  }
};
static MemoryVistor visitor;

}  // namespace tensorflow
}  // namespace byteps
