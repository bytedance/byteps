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

int GpuReducer::copy_d2d(void* dst, const void* src, size_t len) {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, 
    (cudaMemcpyKind) cudaMemcpyDeviceToDevice,
    (cudaStream_t)*_d2d_stream));
  CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
#else
  BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif // BYTEPS_BUILDING_CUDA == 1
  return 0;
}

int GpuReducer::copy_h2d(void* dst, const void* src, size_t len) {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, 
    (cudaMemcpyKind) cudaMemcpyHostToDevice,
    (cudaStream_t)*_h2d_stream));
  CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
#else
  BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif // BYTEPS_BUILDING_CUDA == 1
  return 0;
}

int GpuReducer::copy_d2h(void* dst, const void* src, size_t len) {
#if BYTEPS_BUILDING_CUDA == 1
  CUDA_CALL(cudaMemcpyAsync(dst, src, len, 
    (cudaMemcpyKind) cudaMemcpyDeviceToHost,
    (cudaStream_t)*_d2h_stream));
  CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
#else
  BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif // BYTEPS_BUILDING_CUDA == 1
  return 0;
}

}  // namespace common
}  // namespace byteps

