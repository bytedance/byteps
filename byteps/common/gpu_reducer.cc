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

#include "gpu_reducer.h"

namespace byteps {
namespace common {

int GpuReducer::copy(void* dst, bool to_gpu, const void* src, bool from_gpu,
                     size_t len, bool async) {
  BPS_CHECK(to_gpu || from_gpu) << to_gpu << " " << from_gpu;
  if (to_gpu && from_gpu) {
    return copy_d2d(dst, src, len, async);
  } else if (to_gpu) {
    return copy_h2d(dst, src, len, async);
  } else {
    return copy_d2h(dst, src, len, async);
  }
}

int GpuReducer::copy_d2d(void* dst, const void* src, size_t len, bool async) {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, 
    (cudaMemcpyKind) cudaMemcpyDeviceToDevice,
    (cudaStream_t)*_d2d_stream));
  if (!async) {
    CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
  }
#else
  CUDA_BUILD_ERROR();
#endif // BYTEPS_BUILDING_CUDA == 1
  return 0;
}

int GpuReducer::copy_h2d(void* dst, const void* src, size_t len, bool async) {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, 
    (cudaMemcpyKind) cudaMemcpyHostToDevice,
    (cudaStream_t)*_h2d_stream));
  if (!async) {
    CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
  }
#else
  CUDA_BUILD_ERROR();
#endif // BYTEPS_BUILDING_CUDA == 1
  return 0;
}

int GpuReducer::copy_d2h(void* dst, const void* src, size_t len, bool async) {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, 
    (cudaMemcpyKind) cudaMemcpyDeviceToHost,
    (cudaStream_t)*_d2h_stream));
  if (!async) {
    CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
  }
#else
  CUDA_BUILD_ERROR();
#endif // BYTEPS_BUILDING_CUDA == 1
  return 0;
}

void GpuReducer::sync_h2d() {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
#else
  CUDA_BUILD_ERROR();
#endif // BYTEPS_BUILDING_CUDA == 1
}

void GpuReducer::sync_d2h() {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
#else
  CUDA_BUILD_ERROR();
#endif // BYTEPS_BUILDING_CUDA == 1
}

void GpuReducer::sync_d2d() {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
#else
  CUDA_BUILD_ERROR();
#endif // BYTEPS_BUILDING_CUDA == 1
}

}  // namespace common
}  // namespace byteps

