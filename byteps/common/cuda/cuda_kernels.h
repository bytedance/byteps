// Copyright (C) 2020 NVIDIA CORPORATION. All rights reserved.
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

#ifndef BYTEPS_CUDA_KERNELS_H
#define BYTEPS_CUDA_KERNELS_H

#include <cuda_runtime.h>

#include "../common.h"
#include "../logging.h"

#define BATCHED_D2D_CAPACITY 160
#define BATCHED_D2D_PADDING 16
#define BATCHED_ZO_CAPACITY 160
#define BATCHED_ZO_PADDING 16

namespace byteps {
namespace common {

struct BatchedD2DParams {
  void* out[BATCHED_D2D_CAPACITY];
  void* in[BATCHED_D2D_CAPACITY];
  size_t sizes[BATCHED_D2D_CAPACITY];
};

struct BatchedZOParams {
  void* out[BATCHED_ZO_CAPACITY];
  size_t sizes[BATCHED_ZO_CAPACITY];
};

// Performs a batched d2d memcopy
void BatchedD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, cudaStream_t stream);

// batched zero out of buffers
void BatchedZeroOutCudaImpl(BatchedZOParams& params, int num_zero_out, cudaStream_t stream);

// Scales buffer by scalar
void ScaleBufferCudaImpl(const void* fused_input_data, void* buffer_data, const int64_t num_elements,
                         double scale_factor, DataType dtype, cudaStream_t stream);

void BatchedScaledD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, double scale_factor,
                                    DataType dtype, cudaStream_t stream);

class CudaReducer {
 public:
  CudaReducer(int block_num, int thread_num) :
      _kernel_block_num(block_num), _kernel_thread_num(thread_num) {
    BPS_LOG(DEBUG) << "Create Cuda Reducer with <block_num, thread_num>: " 
        << "<" << _kernel_block_num << ", " << _kernel_thread_num << ">";
  };

  ~CudaReducer() {
    if (_stream) {
      CUDA_CALL(cudaStreamSynchronize(*_stream));
      CUDA_CALL(cudaStreamDestroy(*_stream));
      free(_stream);
      _stream = nullptr;
    }
    BPS_LOG(TRACE) << "Clear CudaReducer";
  } 

  DataType GetDataType(int dtype) { return static_cast<DataType>(dtype); }

  void CopyD2D(void* dst, void* src, size_t len, bool sync);

  void CopyD2DAsync(void* dst, void* src, size_t len, cudaStream_t* stream);

  void Sum(void* dst, const void* src1, const void* src2, size_t len, DataType dtype, bool sync);

  void Sum(void* dst, const void* src, size_t len, DataType dtype, bool sync) {
    Sum(dst, dst, src, len, dtype, sync);  
  }

  void SumAsync(void* dst, const void* src, size_t len, DataType dtype, cudaStream_t* stream) {
    SumAsync(dst, dst, src, len, dtype, stream);
  }

  void SumAsync(void* dst, const void* src1, const void* src2, size_t len, DataType dtype, cudaStream_t* stream);

  void Sync() { CUDA_CALL(cudaStreamSynchronize(*_stream)); }
  
 private:
  int _kernel_block_num;
  int _kernel_thread_num;
  cudaStream_t* _stream = nullptr;

  void InitStream() {
    int greatest_priority;
    _stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    cudaError_t e1 = cudaDeviceGetStreamPriorityRange(
                        NULL, &greatest_priority);
    BPS_CHECK(e1 == cudaSuccess 
              || e1 == cudaErrorCudartUnloading) << "CUDA: " << cudaGetErrorString(e1);  
    cudaError_t e2 = cudaStreamCreateWithPriority(_stream, 
                        cudaStreamNonBlocking, greatest_priority);
    BPS_CHECK(e2 == cudaSuccess 
              || e2 == cudaErrorCudartUnloading) << "CUDA: " << cudaGetErrorString(e2);   
    CUDA_CALL(cudaStreamSynchronize(*_stream));
  }
};


} // namespace common
} // namespace byteps

#endif // BYTEPS_CUDA_KERNELS_H
