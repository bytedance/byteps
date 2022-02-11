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

#include "cuda_kernels.h"

#include <stdexcept>
#include <cuda_fp16.h>



namespace byteps {
namespace common {

// begin of DC cuda kernel section
// template<bool is_half, typename T>
// __device__ float to_float(T& val){
//   if (is_half){
//     return __half2float(val);
//   } else {
//     return val;
//   }
// }

// template<bool is_half, typename T>
// __device__ T to_type(double& val) {
//   float v = static_cast<float>(val);
//   if (is_half){
//     return __float2half(v);
//   } else {
//     return v;
//   }
// }

template<typename T>
__global__ void byte_dc_adam_kernel(DCAdamParams dcadam_params) {

    T *param      = reinterpret_cast<T**>(dcadam_params.params)[blockIdx.x];
    T *grad       = reinterpret_cast<T**>(dcadam_params.grads)[blockIdx.x];
    T *prev_param = reinterpret_cast<T**>(dcadam_params.prev_params)[blockIdx.x];
    int64_t len   = dcadam_params.sizes[blockIdx.x];
    float lambda  = dcadam_params.lambda;

    float *exp_avgs     = dcadam_params.exp_avgs[blockIdx.x];
    float *exp_avg_sqs  = dcadam_params.exp_avg_sqs[blockIdx.x];
    int64_t step        = dcadam_params.steps[blockIdx.x];
    float lr            = dcadam_params.lr;
    float eps           = dcadam_params.eps;
    float weight_decay  = dcadam_params.weight_decay;
    float beta1         = dcadam_params.beta1;
    float beta2         = dcadam_params.beta2;

    int32_t tid = threadIdx.x;

    for (int32_t i = blockIdx.y * blockDim.x + tid; i < len; i += blockDim.x * gridDim.y) {
      float p_val           = static_cast<float>(param[i]);
      float g_val           = static_cast<float>(grad[i]);
      float prev_p_val      = static_cast<float>(prev_param[i]);
      float exp_avg_val     = exp_avgs[i];
      float exp_avg_sq_val  = exp_avg_sqs[i];

      prev_param[i]   = static_cast<T>(p_val);          // store current param into prev_param

      // delay compensation using diagonal approximation
      g_val += lambda * g_val * g_val * (p_val - prev_p_val);

      // adam implementation
      exp_avg_val = exp_avg_val * beta1 + (1 - beta1) * g_val;
      exp_avg_sq_val = exp_avg_sq_val * beta2 + (1 - beta2) * g_val * g_val;
      float denorm = sqrt(exp_avg_sq_val) + eps;
      float bias_correction1 = 1 - pow(beta1, step);
      float bias_correction2 = 1 - powf(beta2, step);
      float step_size = lr * sqrt(bias_correction2) / bias_correction1;
      if (weight_decay != 0.0) {
        p_val -= weight_decay * lr * p_val;
      }
      p_val -= step_size * exp_avg_val / denorm;

      param[i]        = static_cast<T>(p_val);
      exp_avgs[i]     = exp_avg_val;
      exp_avg_sqs[i]  = exp_avg_sq_val;
      grad[i]         = static_cast<T>(g_val);          // store delay compensated grad
    }
}


template<typename T>
__global__ void byte_dc_cuda_kernel(DCParams dc_params) {

    T *param      = reinterpret_cast<T**>(dc_params.params)[blockIdx.x];
    T *grad       = reinterpret_cast<T**>(dc_params.grads)[blockIdx.x];
    T *prev_param = reinterpret_cast<T**>(dc_params.prev_params)[blockIdx.x];
    int64_t len   = dc_params.sizes[blockIdx.x];
    float lambda = dc_params.lambda;

    int32_t tid   = threadIdx.x;

    for (int32_t i = blockIdx.y * blockDim.x + tid; i < len; i += blockDim.x * gridDim.y) {
      float p_val      = static_cast<float>(param[i]);
      float g_val      = static_cast<float>(grad[i]);
      float prev_p_val = static_cast<float>(prev_param[i]);

      // delay compensation using diagonal approximation
      g_val += lambda * g_val * g_val * (p_val - prev_p_val);

      prev_param[i] = static_cast<T>(p_val);          // store current param into prev_param
      grad[i]       = static_cast<T>(g_val);          // store delay compensated grad
    }
}

// template <typename T>
void DCCudaImpl(DCParams& dc_params, int count, cudaStream_t stream, DataType data_type){
  // bool is_half = (data == DataType::BYTEPS_FLOAT16) ? true : false;
  // byte_dc_cuda_kernel<half><<<dim3(count, DC_GRID_SIZE, 1), DC_BLOCK_SIZE, 0, stream>>>(dc_params);
  if (data_type == DataType::BYTEPS_FLOAT16){
    byte_dc_cuda_kernel<half><<<dim3(count, DC_GRID_SIZE, 1), DC_BLOCK_SIZE, 0, stream>>>(dc_params);
  } else {
    byte_dc_cuda_kernel<float><<<dim3(count, DC_GRID_SIZE, 1), DC_BLOCK_SIZE, 0, stream>>>(dc_params);
  }
}

void DCAdamCudaWrapper(DCAdamParams& dcadam_params, size_t count, cudaStream_t stream, DataType data_type){
  if (data_type == DataType::BYTEPS_FLOAT16){
    byte_dc_adam_kernel<half><<<dim3(count, DC_GRID_SIZE, 1), DC_BLOCK_SIZE, 0, stream>>>(dcadam_params);
  } else {
    byte_dc_adam_kernel<float><<<dim3(count, DC_GRID_SIZE, 1), DC_BLOCK_SIZE, 0, stream>>>(dcadam_params);
  }
}


template<typename T, int blocks_per_copy>
__device__ void batched_memcpy_d(size_t idx, const void* in, void* out, size_t size) {

  const T* input = reinterpret_cast<const T *>(in);
  T* output = reinterpret_cast<T *>(out);
  const size_t num_elements = size / sizeof(T);

  for (size_t i = idx; i < num_elements; i += blockDim.x * blocks_per_copy) {
    output[i] = input[i];
  }

  // Deal with any remaining bytes
  size_t remainder = size % sizeof(T);
  if (remainder > 0 && idx < remainder) {
    const unsigned char* input_r = reinterpret_cast<const unsigned char *>(input + num_elements);
    unsigned char* output_r = reinterpret_cast<unsigned char *>(output + num_elements);
    output_r[idx] = input_r[idx];
  }
}

template<int blocks_per_copy>
__global__ void batched_memcpy_k(BatchedD2DParams params) {
  const size_t idx = blockDim.x * (blockIdx.x % blocks_per_copy) + threadIdx.x;

  const size_t size = params.sizes[blockIdx.x / blocks_per_copy];
  const void* input = params.in[blockIdx.x / blocks_per_copy];
  void* output = params.out[blockIdx.x / blocks_per_copy];

  // Check alignment relative to 16 bytes
  size_t align_in = reinterpret_cast<size_t>(input) % BATCHED_D2D_PADDING;
  size_t align_out = reinterpret_cast<size_t>(output) % BATCHED_D2D_PADDING;

  // Select load/store size based on the misaligned buffer
  size_t align = (align_out == 0) ? align_in : align_out;
  if (align_in && align_out) {
    // If both are misaligned, use unsigned char (this should not occur
    // as fusion buffer locations should be aligned by applying BATCHED_D2D_PADDING
    // during construction.)
    align = 1;
  }

  if (align % 16 == 0) {
    batched_memcpy_d<ulonglong2, blocks_per_copy>(idx, input, output, size);
  } else if (align % 8 == 0) {
    batched_memcpy_d<unsigned long long, blocks_per_copy>(idx, input, output, size);
  } else if (align % 4 == 0) {
    batched_memcpy_d<unsigned int, blocks_per_copy>(idx, input, output, size);
  } else if (align % 2 == 0) {
    batched_memcpy_d<unsigned short, blocks_per_copy>(idx, input, output, size);
  } else {
    batched_memcpy_d<unsigned char, blocks_per_copy>(idx, input, output, size);
  }
}

#define NTHREADS_D2D_KERNEL 1024
#define BLOCKS_PER_COPY_D2D_KERNEL 8
void BatchedD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, cudaStream_t stream)
{
   batched_memcpy_k<BLOCKS_PER_COPY_D2D_KERNEL><<<num_copies * BLOCKS_PER_COPY_D2D_KERNEL,
                                                  NTHREADS_D2D_KERNEL, 0, stream>>>(params);
}

template<typename T, int blocks_per_zero_out>
__device__ void batched_zero_out_d(size_t idx, void* buff, size_t size) {

  T* output = reinterpret_cast<T *>(buff);
  const size_t num_elements = size / sizeof(T);

  for (size_t i = idx; i < num_elements; i += blockDim.x * blocks_per_zero_out) {
    output[i] = 0;
  }

  // Deal with any remaining bytes
  size_t remainder = size % sizeof(T);
  if (remainder > 0 && idx < remainder) {
    unsigned char* output_r = reinterpret_cast<unsigned char *>(output + num_elements);
    output_r[idx] = 0;
  }
}

template<typename T, int blocks_per_zero_out>
__device__ void batched_zero_out_ull2_d(size_t idx, void* buff, size_t size) {

  T* output = reinterpret_cast<T *>(buff);
  const size_t num_elements = size / sizeof(T);
  ulonglong2 my_zero = make_ulonglong2(0, 0);

  for (size_t i = idx; i < num_elements; i += blockDim.x * blocks_per_zero_out) {
    output[i] = my_zero;
  }

  // Deal with any remaining bytes
  size_t remainder = size % sizeof(T);
  if (remainder > 0 && idx < remainder) {
    unsigned char* output_r = reinterpret_cast<unsigned char *>(output + num_elements);
    output_r[idx] = 0;
  }
}

template<int blocks_per_zero_out>
__global__ void batched_zero_out_k(BatchedZOParams params) {
  const size_t idx = blockDim.x * (blockIdx.x % blocks_per_zero_out) + threadIdx.x;

  const size_t size = params.sizes[blockIdx.x / blocks_per_zero_out];
  void* output = params.out[blockIdx.x / blocks_per_zero_out];

  // Check alignment relative to 16 bytes
  size_t align_out = reinterpret_cast<size_t>(output) % BATCHED_ZO_PADDING;

  // Select load/store size based on the misaligned buffer
  size_t align = align_out;
  ulonglong2 my_zero_ull2 = make_ulonglong2(0, 0);


  if (align % 16 == 0) {
    batched_zero_out_ull2_d<ulonglong2, blocks_per_zero_out>(idx, output, size);
  } else if (align % 8 == 0) {
    batched_zero_out_d<unsigned long long, blocks_per_zero_out>(idx, output, size);
  } else if (align % 4 == 0) {
    batched_zero_out_d<unsigned int, blocks_per_zero_out>(idx, output, size);
  } else if (align % 2 == 0) {
    batched_zero_out_d<unsigned short, blocks_per_zero_out>(idx, output, size);
  } else {
    batched_zero_out_d<unsigned char, blocks_per_zero_out>(idx, output, size);
  }
}

#define NTHREADS_ZO_KERNEL 1024
#define BLOCKS_PER_ZERO_OUT_KERNEL 8
void BatchedZeroOutCudaImpl(BatchedZOParams& params, int num_zero_out, cudaStream_t stream)
{
   batched_zero_out_k<BLOCKS_PER_ZERO_OUT_KERNEL><<<num_zero_out * BLOCKS_PER_ZERO_OUT_KERNEL,
                                                  NTHREADS_ZO_KERNEL, 0, stream>>>(params);
}

template<typename T, typename TS>
__global__ void scale_buffer_k(const T* input, T* output, int64_t num_elements, const TS scale_factor) {

  const size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;

  for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
    output[i] = scale_factor * input[i];
  }
}

// Specialization for half2
__global__ void scale_buffer_half2_k(const __half* input, __half* output, int64_t num_elements, const __half scale_factor) {

  const size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;

#if __CUDA_ARCH__ > 530
  const __half2* input_h2 = reinterpret_cast<const __half2 *>(input);
  __half2* output_h2 = reinterpret_cast<__half2 *>(output);
  const __half2 scale_factor_h2 = __halves2half2(scale_factor, scale_factor);

  for (size_t i = idx; i < num_elements / 2; i += gridDim.x * blockDim.x) {
    output_h2[i] = __hmul2(scale_factor_h2, input_h2[i]);
  }

  // Deal with last element if num_elements is odd
  if (idx == 0 && num_elements % 2) {
    output[num_elements - 1] = __hmul(scale_factor, input[num_elements - 1]);
  }
#else
  for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
    output[i] = __float2half(__half2float(scale_factor) * __half2float(input[i]));
  }
#endif
}

// Specialization for architectures without __half compute
template<>
__global__ void scale_buffer_k(const __half* input, __half* output, int64_t num_elements, const __half scale_factor) {

  const size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;

#if __CUDA_ARCH__ > 530
  for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
    output[i] = scale_factor * input[i];
  }
#else
  for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
    output[i] = __float2half(__half2float(scale_factor) * __half2float(input[i]));
  }
#endif
}

#define NTHREADS_SCALE_BUFFER_KERNEL 512
void ScaleBufferCudaImpl(const void* fused_input_data, void* buffer_data, const int64_t num_elements, double scale_factor,
                         DataType dtype, cudaStream_t stream) {
  const int64_t blocks = (num_elements + NTHREADS_SCALE_BUFFER_KERNEL - 1) / NTHREADS_SCALE_BUFFER_KERNEL;
  const int threads = NTHREADS_SCALE_BUFFER_KERNEL;
  switch (dtype) {
    case BYTEPS_UINT8:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const uint8_t*) fused_input_data, (uint8_t*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    case BYTEPS_INT8:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const int8_t*) fused_input_data, (int8_t*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    case BYTEPS_INT32:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const int32_t*) fused_input_data, (int32_t*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    case BYTEPS_INT64:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const int64_t*) fused_input_data, (int64_t*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    case BYTEPS_FLOAT16:
    {
      __half scale_factor_half = __float2half((float) scale_factor);
      if ((size_t) fused_input_data % 4 == 0 && (size_t) buffer_data % 4 == 0) {
        // If alignment allows, use half2 specialized kernel
        int64_t num_elements_h2 = (num_elements + 1) / 2;
        int64_t blocks_h2 = (num_elements_h2 + NTHREADS_SCALE_BUFFER_KERNEL - 1) / NTHREADS_SCALE_BUFFER_KERNEL;
        scale_buffer_half2_k<<<blocks_h2, threads, 0, stream>>>((const __half*) fused_input_data, (__half*) buffer_data,
                                                          num_elements, scale_factor_half);
      } else {
        scale_buffer_k<<<blocks, threads, 0, stream>>>((const __half*) fused_input_data, (__half*) buffer_data,
                                                       num_elements, scale_factor_half);
     }
      break;
    }
    case BYTEPS_FLOAT32:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const float*) fused_input_data, (float*) buffer_data,
                                                     num_elements, (float) scale_factor);
      break;
    case BYTEPS_FLOAT64:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const double*) fused_input_data, (double*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                             " not supported by ScaleBufferCudaImpl.");
  }
}

template<typename TL, int blocks_per_copy, typename T, typename TS>
__device__ void batched_scaled_memcpy_d(size_t idx, const T* input, T* output, size_t size, const TS scale_factor) {

  const int64_t num_words = size / sizeof(TL);
  const TL* read_ptr = reinterpret_cast<const TL*>(input);
  TL* write_ptr = reinterpret_cast<TL*>(output);
  for (size_t i = idx; i < num_words; i += blockDim.x * blocks_per_copy) {
    // Load word
    TL word = read_ptr[i];
    T* val = reinterpret_cast<T*>(&word);

    // Scale elements in word
    for (int j = 0; j < sizeof(TL) / sizeof(T); ++j) {
      val[j] *= scale_factor;
    }

    // Write word
    write_ptr[i] = word;
  }

  // Deal with any remaining elements
  size_t remainder = (size % sizeof(TL)) / sizeof(T);
  if (remainder > 0 && idx < remainder) {
    const T* input_r = reinterpret_cast<const T*>(read_ptr + num_words);
    T* output_r = reinterpret_cast<T*>(write_ptr + num_words);
    output_r[idx] = scale_factor * input_r[idx];
  }
}

// Specialization for architectures without __half compute
template<typename TL, int blocks_per_copy>
__device__ void batched_scaled_memcpy_d(size_t idx, const __half* input, __half* output, size_t size, const __half scale_factor) {

  const int64_t num_words = size / sizeof(TL);
  const TL* read_ptr = reinterpret_cast<const TL*>(input);
  TL* write_ptr = reinterpret_cast<TL*>(output);
  for (size_t i = idx; i < num_words; i += blockDim.x * blocks_per_copy) {
    // Load word
    TL word = read_ptr[i];
    __half* val = reinterpret_cast<__half*>(&word);

    // Scale elements in word
    for (int j = 0; j < sizeof(TL) / sizeof(__half); ++j) {
#if __CUDA_ARCH__ > 530
      val[j] *= scale_factor;
#else
      val[j] = __float2half(__half2float(scale_factor) * __half2float(val[j]));
#endif
    }

    // Write word
    write_ptr[i] = word;
  }

  // Deal with any remaining elements
  size_t remainder = (size % sizeof(TL)) / sizeof(__half);
  if (remainder > 0 && idx < remainder) {
    const __half* input_r = reinterpret_cast<const __half*>(read_ptr + num_words);
    __half* output_r = reinterpret_cast<__half*>(write_ptr + num_words);
#if __CUDA_ARCH__ > 530
    output_r[idx] = scale_factor * input_r[idx];
#else
    output_r[idx] = __float2half(__half2float(scale_factor) * __half2float(input_r[idx]));
#endif
  }
}

template<typename T, int blocks_per_copy, typename TS>
__global__ void batched_scaled_memcpy_k(BatchedD2DParams params, TS scale_factor) {
  const size_t idx = blockDim.x * (blockIdx.x % blocks_per_copy) + threadIdx.x;

  const size_t size = params.sizes[blockIdx.x / blocks_per_copy];
  const T* input = reinterpret_cast<const T*>(params.in[blockIdx.x / blocks_per_copy]);
  T* output = reinterpret_cast<T*>(params.out[blockIdx.x / blocks_per_copy]);

  // Check alignment relative to 16 bytes
  size_t align_in = reinterpret_cast<size_t>(input) % BATCHED_D2D_PADDING;
  size_t align_out = reinterpret_cast<size_t>(output) % BATCHED_D2D_PADDING;

  // Select load/store size based on the misaligned buffer
  size_t align = (align_out == 0) ? align_in : align_out;
  if (align_in && align_out) {

    // If both are misaligned, use datatype size
    align = sizeof(T);
  }

  if (align % 16 == 0) {
    batched_scaled_memcpy_d<ulonglong2, blocks_per_copy>(idx, input, output, size, scale_factor);
  } else if (align % 8 == 0) {
    batched_scaled_memcpy_d<unsigned long long, blocks_per_copy>(idx, input, output, size, scale_factor);
  } else if (align % 4 == 0) {
    batched_scaled_memcpy_d<unsigned int, blocks_per_copy>(idx, input, output, size, scale_factor);
  } else if (align % 2 == 0) {
    batched_scaled_memcpy_d<unsigned short, blocks_per_copy>(idx, input, output, size, scale_factor);
  } else {
    batched_scaled_memcpy_d<unsigned char, blocks_per_copy>(idx, input, output, size, scale_factor);
  }
}

void BatchedScaledD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, double scale_factor,
                                    DataType dtype, cudaStream_t stream) {
  const int64_t blocks = num_copies * BLOCKS_PER_COPY_D2D_KERNEL;
  const int threads = NTHREADS_D2D_KERNEL;
  switch (dtype) {
   case BYTEPS_UINT8:
     batched_scaled_memcpy_k<uint8_t, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   case BYTEPS_INT8:
     batched_scaled_memcpy_k<int8_t, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   case BYTEPS_INT32:
     batched_scaled_memcpy_k<int32_t, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   case BYTEPS_INT64:
     batched_scaled_memcpy_k<int64_t, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   case BYTEPS_FLOAT16: {
     __half scale_factor_half = __float2half((float) scale_factor);
     batched_scaled_memcpy_k<__half, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor_half);
     break;
   }
   case BYTEPS_FLOAT32:
     batched_scaled_memcpy_k<float, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, (float) scale_factor);
     break;
   case BYTEPS_FLOAT64:
     batched_scaled_memcpy_k<double, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   default:
     throw std::logic_error("Type " + DataType_Name(dtype) +
                            " not supported by BatchedScaledD2DMemcpyCudaImpl.");
  }
}

template <typename T>
__global__ void _SumKernel(void* dst, const void* src1, const void* src2, size_t len) {
  T* d = (T*) dst;
  T* s1 = (T*) src1;
  T* s2 = (T*) src2;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;
  for(; tid < len; tid += total_threads) {
    d[tid] = __ldg(&s1[tid]) + __ldg(&s2[tid]);
  }
}

__global__ void _SumKernelFloat16(void* dst, const void* src1, const void* src2, size_t len) {
  __half* d = (__half*) dst;
  __half* s1 = (__half*) src1;
  __half* s2 = (__half*) src2;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;
  for(; tid < len; tid += total_threads) {
    d[tid] = __hadd(__ldg(&s1[tid]), __ldg(&s2[tid]));
  }
}

__global__ void _CopyKernel(void* dst, const void* src, size_t len) {
  char* d = (char*) dst;
  char* s = (char*) src;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;
  for(; tid < len; tid += total_threads) {
    d[tid] = __ldg(&s[tid]);
  }
}

void CudaReducer::Sum(void* dst, const void* src1, const void* src2, size_t len, DataType dtype, bool sync) {
  if (!_stream) InitStream();
  switch (dtype) {
    case BYTEPS_FLOAT16:
      _SumKernelFloat16<<< _kernel_block_num, _kernel_thread_num, 0, *_stream >>>(dst, src1, src2, len/2);
      break;
    case BYTEPS_FLOAT32:
      _SumKernel<float><<< _kernel_block_num, _kernel_thread_num, 0, *_stream >>>(dst, src1, src2, len/sizeof(float));
      break;
    case BYTEPS_FLOAT64:
      _SumKernel<double><<< _kernel_block_num, _kernel_thread_num, 0, *_stream >>>(dst, src1, src2, len/sizeof(double));
      break;
    case BYTEPS_INT32:
      _SumKernel<int><<< _kernel_block_num, _kernel_thread_num, 0, *_stream >>>(dst, src1, src2, len/sizeof(int));
      break;
    case BYTEPS_INT64:
      _SumKernel<long><<< _kernel_block_num, _kernel_thread_num, 0, *_stream >>>(dst, src1, src2, len/sizeof(long));
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type for Cuda Reducer: " << dtype;
  }
  if (sync) CUDA_CALL(cudaStreamSynchronize(*_stream));
}

void CudaReducer::SumAsync(void* dst, const void* src1, const void* src2, size_t len, DataType dtype, cudaStream_t* stream) {
  switch (dtype) {
    case BYTEPS_FLOAT16:
      _SumKernelFloat16<<< _kernel_block_num, _kernel_thread_num, 0, *stream >>>(dst, src1, src2, len/2);
      break;
    case BYTEPS_FLOAT32:
      _SumKernel<float><<< _kernel_block_num, _kernel_thread_num, 0, *stream >>>(dst, src1, src2, len/sizeof(float));
      break;
    case BYTEPS_FLOAT64:
      _SumKernel<double><<< _kernel_block_num, _kernel_thread_num, 0, *stream >>>(dst, src1, src2, len/sizeof(double));
      break;
    case BYTEPS_INT32:
      _SumKernel<int><<< _kernel_block_num, _kernel_thread_num, 0, *stream >>>(dst, src1, src2, len/sizeof(int));
      break;
    case BYTEPS_INT64:
      _SumKernel<long><<< _kernel_block_num, _kernel_thread_num, 0, *stream >>>(dst, src1, src2, len/sizeof(long));
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type for Cuda Reducer: " << dtype;
  }
}

void CudaReducer::CopyD2D(void* dst, void* src, size_t len, bool sync) {
  if (!_stream) InitStream();
  _CopyKernel<<< _kernel_block_num, _kernel_thread_num, 0, *_stream >>>(dst, src, len);
  if (sync) CUDA_CALL(cudaStreamSynchronize(*_stream));
}

void CudaReducer::CopyD2DAsync(void* dst, void* src, size_t len, cudaStream_t* stream) {
  _CopyKernel<<< _kernel_block_num, _kernel_thread_num, 0, *stream >>>(dst, src, len);
}

} // namespace common
} // namespace byteps

