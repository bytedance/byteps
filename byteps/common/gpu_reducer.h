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

#ifndef BYTEPS_GPU_REDUCER_H
#define BYTEPS_GPU_REDUCER_H

#if BYTEPS_BUILDING_CUDA == 1
#include <cuda_runtime.h>
#endif // BYTEPS_BUILDING_CUDA == 1

#include "common.h"
#include "logging.h"

namespace byteps {
namespace common {

class GpuReducer {
 public:
  GpuReducer() { InitStream(); }
  int copy(void* dst, bool to_gpu, const void* src, bool from_gpu,
           size_t len, bool async);
  int copy_h2d(void* dst, const void* src, size_t len, bool async);
  int copy_d2h(void* dst, const void* src, size_t len, bool async);
  int copy_d2d(void* dst, const void* src, size_t len, bool async);

  void sync_h2d();
  void sync_d2h();
  void sync_d2d();

#if BYTEPS_BUILDING_CUDA == 1
  ~GpuReducer() {
    if (_h2d_stream) {
      CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
      free(_h2d_stream);
    }
    if (_d2h_stream) {
      CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
      free(_d2h_stream);
    }
    if (_d2d_stream) {
      CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
      free(_d2d_stream);
    }
  }
#endif // BYTEPS_BUILDING_CUDA == 1

 private:
#if BYTEPS_BUILDING_CUDA == 1
  cudaStream_t* _h2d_stream = NULL;
  cudaStream_t* _d2h_stream = NULL;
  cudaStream_t* _d2d_stream = NULL;

  void InitStream() {
    _h2d_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _d2h_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _d2d_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    CUDA_CALL(cudaStreamCreateWithFlags(_h2d_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_d2h_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_d2d_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamSynchronize(*_h2d_stream));
    CUDA_CALL(cudaStreamSynchronize(*_d2h_stream));
    CUDA_CALL(cudaStreamSynchronize(*_d2d_stream));
  }
#else
  void InitStream() {}
#endif // BYTEPS_BUILDING_CUDA == 1
};


}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_GPU_REDUCER_H
