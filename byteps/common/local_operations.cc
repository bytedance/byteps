// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Copyright 2021 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include "local_operations.h"
#if BYTEPS_BUILDING_CUDA == 1
#include <cuda_runtime.h>
#endif

#if BYTEPS_BUILDING_CUDA == 1
#include "cuda/cuda_kernels.h"
#endif

namespace byteps {
namespace common {

#if BYTEPS_BUILDING_CUDA == 1
void MemcpyInFusionBuffer(std::vector<std::shared_ptr<Tensor>>& src, char *dst, cudaStream_t stream) {
  if (true) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    for (auto& e : src) {
      void* buffer_data_at_offset = (uint8_t*)dst + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = (void*) e->data();
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e->size();

      offset += e->size();
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int) src.size()) {
        // Perform batched d2d memcpy
        BatchedD2DMemcpyCudaImpl(d2d_params, count, stream);
        count = 0;
      }
    }
  }
}

void MemcpyOutFusionBuffer(char *src, std::vector<std::shared_ptr<Tensor>>& dst, cudaStream_t stream) {
  if (true) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    for (auto& e : dst) {
      void* buffer_data_at_offset = (uint8_t*)src + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = (void*) e->data();
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e->size();

      offset += e->size();
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int) dst.size()) {
        // Perform batched d2d memcpy
        BatchedD2DMemcpyCudaImpl(d2d_params, count, stream);
        count = 0;
      }
    }
  }
}

void ZeroOutTensors(std::vector<std::shared_ptr<Tensor>>& dst, cudaStream_t stream) {
  if (true) {
    int idx = 0;
    int count = 0;

    BatchedZOParams zo_params;
    for (auto& e : dst) {
      // Set input/output pointers and sizes
      zo_params.out[idx % BATCHED_ZO_CAPACITY] = (void*) e->data();
      zo_params.sizes[idx % BATCHED_ZO_CAPACITY] = e->size();

      idx++;
      count++;

      if (idx % BATCHED_ZO_CAPACITY == 0 || idx == (int) dst.size()) {
        // Perform batched d2d memcpy
        BatchedZeroOutCudaImpl(zo_params, count, stream);
        count = 0;
      }
    }
  }
}
#endif

}  // namespace common
}  // namespace byteps
