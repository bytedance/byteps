// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef BYTEPS_TORCH_TENSOR_UTIL_H
#define BYTEPS_TORCH_TENSOR_UTIL_H

#include <TH/TH.h>
#include <cassert>

#if HAVE_CUDA
#include <THC/THC.h>
#endif

#include "../common/common.h"
#include "cuda_util.h"

#if HAVE_CUDA
extern THCState* state;
#endif

namespace byteps {
namespace torch {

using namespace byteps::common;

// TH<xxx>Tensor are all aliased to THTensor as of PyTorch 0.4.1, so we need
// an additional template parameter to distinguish between them.
class TensorUtil {
public:
  template <DataType DT, DeviceType Dev, class T>
  static const TensorShape GetShape(T* tensor);
  template <DataType DT, DeviceType Dev, class T>
  static const void* GetData(T* tensor);
  template <DataType DT, DeviceType Dev, class T>
  static int64_t GetSize(T* tensor);
  template <DataType DT, DeviceType Dev, class T>
  static int GetDevice(T* tensor);

  template <DataType DT, DeviceType Dev, class T> static T* New(int device);
  template <DataType DT, DeviceType Dev, class T>
  static void Free(T* tensor);
  template <DataType DT, DeviceType Dev, class T>
  static void ResizeNd(T* tensor, int nDimension, int64_t* size,
                       int64_t* stride);
  template <DataType DT, DeviceType Dev, class T>
  static void Copy(T* output, T* tensor);
  template <DataType DT, DeviceType Dev, class T>
  static void DivideTensorInPlace(T* tensor, int value);

#if HAVE_CUDA
  template <DataType DT, class T, class TC>
  static void CopyCPUToCuda(T* cpu, TC* cuda);
  template <DataType DT, class TC, class T>
  static void AsyncCopyCudaToCPU(TC* cuda, T* cpu);
#endif
};

#define TENSOR_UTIL_DEFINE_TYPE_H(BytePSType, DeviceType, THTensor)           \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<BytePSType, DeviceType, THTensor>(   \
      THTensor * tensor);                                                      \
  template <>                                                                  \
  const void* TensorUtil::GetData<BytePSType, DeviceType, THTensor>(          \
      THTensor * tensor);                                                      \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<BytePSType, DeviceType, THTensor>(THTensor *    \
                                                                 tensor);      \
  template <>                                                                  \
  int TensorUtil::GetDevice<BytePSType, DeviceType, THTensor>(THTensor *      \
                                                               tensor);        \
                                                                               \
  template <>                                                                  \
  THTensor* TensorUtil::New<BytePSType, DeviceType, THTensor>(int device);    \
  template <>                                                                  \
  void TensorUtil::Free<BytePSType, DeviceType, THTensor>(THTensor * tensor); \
  template <>                                                                  \
  void TensorUtil::ResizeNd<BytePSType, DeviceType, THTensor>(                \
      THTensor * tensor, int nDimension, int64_t* size, int64_t* stride);      \
  template <>                                                                  \
  void TensorUtil::Copy<BytePSType, DeviceType, THTensor>(THTensor * output,  \
                                                           THTensor * tensor); \
  template <>                                                                  \
  void TensorUtil::DivideTensorInPlace<BytePSType, DeviceType, THTensor>(     \
      THTensor * tensor, int value);

#define TENSOR_UTIL_DEFINE_CPU_TYPE_H(BytePSType, THTensor)                   \
  TENSOR_UTIL_DEFINE_TYPE_H(BytePSType, DeviceType::CPU, THTensor)

#define TENSOR_UTIL_DEFINE_CUDA_TYPE_H(BytePSType, THCTensor, THTensor)       \
  TENSOR_UTIL_DEFINE_TYPE_H(BytePSType, DeviceType::GPU, THCTensor)           \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<BytePSType, THTensor, THCTensor>(            \
      THTensor * cpu, THCTensor * cuda);                                       \
  template <>                                                                  \
  void TensorUtil::AsyncCopyCudaToCPU<BytePSType, THCTensor, THTensor>(       \
      THCTensor * cuda, THTensor * cpu);

#define TENSOR_UTIL_DEFINE_CPU_TYPE(BytePSType, THTensor, THStorage)          \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<BytePSType, DeviceType::CPU,         \
                                         THTensor>(THTensor * tensor) {        \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THTensor##_nDimension(tensor); idx++) {            \
      shape.AddDim(THTensor##_size(tensor, idx));                              \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  const void* TensorUtil::GetData<BytePSType, DeviceType::CPU, THTensor>(     \
      THTensor * tensor) {                                                     \
    return THTensor##_data(tensor);                                            \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<BytePSType, DeviceType::CPU, THTensor>(         \
      THTensor * tensor) {                                                     \
    return (int64_t)(THStorage##_size(THTensor##_storage(tensor)) *            \
                     THStorage##_elementSize());                               \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int TensorUtil::GetDevice<BytePSType, DeviceType::CPU, THTensor>(THTensor * \
                                                                    tensor) {  \
    return CPU_DEVICE_ID;                                                      \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  THTensor* TensorUtil::New<BytePSType, DeviceType::CPU, THTensor>(           \
      int device) {                                                            \
    assert(device == CPU_DEVICE_ID);                                           \
    return THTensor##_new();                                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Free<BytePSType, DeviceType::CPU, THTensor>(THTensor *     \
                                                                tensor) {      \
    THTensor##_free(tensor);                                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<BytePSType, DeviceType::CPU, THTensor>(           \
      THTensor * tensor, int nDimension, int64_t* size, int64_t* stride) {     \
    THTensor##_resizeNd(tensor, nDimension, size, stride);                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<BytePSType, DeviceType::CPU, THTensor>(               \
      THTensor * output, THTensor * tensor) {                                  \
    THTensor##_copy(output, tensor);                                           \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void                                                                         \
  TensorUtil::DivideTensorInPlace<BytePSType, DeviceType::CPU, THTensor>(     \
      THTensor * tensor, int value) {                                          \
    THTensor##_div(tensor, tensor, value);                                     \
  }

#define TENSOR_UTIL_DEFINE_CUDA_TYPE(BytePSType, THCTensor, THTensor,         \
                                     THCStorage)                               \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<BytePSType, DeviceType::GPU,         \
                                         THCTensor>(THCTensor * tensor) {      \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THCTensor##_nDimension(state, tensor); idx++) {    \
      shape.AddDim(THCTensor##_size(state, tensor, idx));                      \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  const void* TensorUtil::GetData<BytePSType, DeviceType::GPU, THCTensor>(    \
      THCTensor * tensor) {                                                    \
    return THCTensor##_data(state, tensor);                                    \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<BytePSType, DeviceType::GPU, THCTensor>(        \
      THCTensor * tensor) {                                                    \
    return (int64_t)(                                                          \
        THCStorage##_size(state, THCTensor##_storage(state, tensor)) *         \
        THCStorage##_elementSize(state));                                      \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int TensorUtil::GetDevice<BytePSType, DeviceType::GPU, THCTensor>(          \
      THCTensor * tensor) {                                                    \
    return THCTensor##_getDevice(state, tensor);                               \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  THCTensor* TensorUtil::New<BytePSType, DeviceType::GPU, THCTensor>(         \
      int device) {                                                            \
    with_device device_context(device);                                        \
    return THCTensor##_new(state);                                             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Free<BytePSType, DeviceType::GPU, THCTensor>(THCTensor *   \
                                                                 tensor) {     \
    THCTensor##_free(state, tensor);                                           \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<BytePSType, DeviceType::GPU, THCTensor>(          \
      THCTensor * tensor, int nDimension, int64_t* size, int64_t* stride) {    \
    with_device device_context(THCTensor##_getDevice(state, tensor));          \
    THCTensor##_resizeNd(state, tensor, nDimension, size, stride);             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<BytePSType, DeviceType::GPU, THCTensor>(              \
      THCTensor * output, THCTensor * tensor) {                                \
    with_device device_context(THCTensor##_getDevice(state, output));          \
    THCTensor##_copy(state, output, tensor);                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void                                                                         \
  TensorUtil::DivideTensorInPlace<BytePSType, DeviceType::GPU, THCTensor>(    \
      THCTensor * tensor, int value) {                                         \
    with_device device_context(THCTensor##_getDevice(state, tensor));          \
    THCTensor##_div(state, tensor, tensor, value);                             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<BytePSType, THTensor, THCTensor>(            \
      THTensor * cpu, THCTensor * cuda) {                                      \
    with_device device_context(THCTensor##_getDevice(state, cuda));            \
    THLongStorage* size = THTensor##_newSizeOf(cpu);                           \
    if (!THCTensor##_isSize(state, cuda, size)) {                              \
      THCTensor##_resize(state, cuda, size, NULL);                             \
    }                                                                          \
    THLongStorage_free(size);                                                  \
    THCTensor##_copyCPU(state, cuda, cpu);                                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::AsyncCopyCudaToCPU<BytePSType, THCTensor, THTensor>(       \
      THCTensor * cuda, THTensor * cpu) {                                      \
    with_device device_context(THCTensor##_getDevice(state, cuda));            \
    THLongStorage* size = THCTensor##_newSizeOf(state, cuda);                  \
    if (!THTensor##_isSize(cpu, size)) {                                       \
      THTensor##_resize(cpu, size, NULL);                                      \
    }                                                                          \
    THLongStorage_free(size);                                                  \
    THTensor##_copyAsyncCuda(state, cpu, cuda);                                \
  }

TENSOR_UTIL_DEFINE_CPU_TYPE_H(DataType::BYTEPS_UINT8, THByteTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(DataType::BYTEPS_INT8, THCharTensor)
// TENSOR_UTIL_DEFINE_CPU_TYPE_H(DataType::BYTEPS_INT16, THShortTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(DataType::BYTEPS_INT32, THIntTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(DataType::BYTEPS_INT64, THLongTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(DataType::BYTEPS_FLOAT32, THFloatTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(DataType::BYTEPS_FLOAT64, THDoubleTensor)

#if HAVE_CUDA
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(DataType::BYTEPS_UINT8, THCudaByteTensor,
                               THByteTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(DataType::BYTEPS_INT8, THCudaCharTensor,
                               THCharTensor)
// TENSOR_UTIL_DEFINE_CUDA_TYPE_H(DataType::BYTEPS_INT16, THCudaShortTensor,
//                               THShortTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(DataType::BYTEPS_INT32, THCudaIntTensor,
                               THIntTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(DataType::BYTEPS_INT64, THCudaLongTensor,
                               THLongTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(DataType::BYTEPS_FLOAT32, THCudaTensor,
                               THFloatTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(DataType::BYTEPS_FLOAT64, THCudaDoubleTensor,
                               THDoubleTensor)
#endif

} // namespace torch
} // namespace byteps

#endif // BYTEPS_TORCH_TENSOR_UTIL_H