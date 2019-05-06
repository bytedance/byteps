// Copyright 2019 ByteDance Inc. All Rights Reserved.
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

#include <chrono>
#include <memory>
#include <thread>

#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "ops.h"
#include "ready_event.h"
#include "tensor_util.h"

namespace byteps {
namespace torch {

static HandleManager handle_manager;

namespace {

std::string GetOpName(const std::string& prefix, char* name, int handle) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

} // namespace

template <DataType DT, DeviceType Dev, class T>
int DoAllreduce(T* tensor, T* output, int average, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = TensorUtil::GetDevice<DT, Dev>(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor<DT, Dev, T>>(tensor);
  auto hvd_context =
      std::make_shared<TorchOpContext<DT, Dev, T>>(device, output);
  auto hvd_output = std::make_shared<TorchTensor<DT, Dev, T>>(output);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, ready_event,
      GetOpName("allreduce", name, handle), device,
      [handle, average, output](const Status& status) {
        if (average) {
          TensorUtil::DivideTensorInPlace<DT, Dev, T>(output, byteps_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
template <DataType DT, class TC, class T>
int DoAllreduceCudaOnCPU(TC* tensor, TC* output, int average, char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice<DT, DeviceType::GPU>(tensor);
  auto hvd_cpu_buffer =
      std::make_shared<TorchTemporaryBuffer<DT, DeviceType::CPU, T>>(
          CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU<DT>(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context = std::make_shared<TorchOpContext<DT, DeviceType::CPU, T>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      GetOpName("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, average, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<DT>(hvd_cpu_buffer->tensor(), output);
        if (average) {
          TensorUtil::DivideTensorInPlace<DT, DeviceType::GPU>(output,
                                                               byteps_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

#define PUSHPULL(torch_Tensor, BytePSType, DeviceType, THTensor)             \
  extern "C" int byteps_torch_allreduce_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, int average, char* name) {           \
    return DoAllreduce<BytePSType, DeviceType>(tensor, output, average,       \
                                                name);                         \
  }

PUSHPULL(torch_IntTensor, DataType::BYTEPS_INT32, DeviceType::CPU,
          THIntTensor)
PUSHPULL(torch_LongTensor, DataType::BYTEPS_INT64, DeviceType::CPU,
          THLongTensor)
PUSHPULL(torch_FloatTensor, DataType::BYTEPS_FLOAT32, DeviceType::CPU,
          THFloatTensor)
PUSHPULL(torch_DoubleTensor, DataType::BYTEPS_FLOAT64, DeviceType::CPU,
          THDoubleTensor)

#if BYTEPS_GPU_PUSHPULL
PUSHPULL(torch_cuda_IntTensor, DataType::BYTEPS_INT32, DeviceType::GPU,
          THCudaIntTensor)
PUSHPULL(torch_cuda_LongTensor, DataType::BYTEPS_INT64, DeviceType::GPU,
          THCudaLongTensor)
PUSHPULL(torch_cuda_FloatTensor, DataType::BYTEPS_FLOAT32, DeviceType::GPU,
          THCudaTensor)
PUSHPULL(torch_cuda_DoubleTensor, DataType::BYTEPS_FLOAT64,
          DeviceType::GPU, THCudaDoubleTensor)
#endif

#define PUSHPULL_CUDA_ON_CPU(torch_Tensor, BytePSType, THCTensor, THTensor)  \
  extern "C" int byteps_torch_allreduce_async_##torch_Tensor(                 \
      THCTensor* tensor, THCTensor* output, int average, char* name) {         \
    return DoAllreduceCudaOnCPU<BytePSType, THCTensor, THTensor>(             \
        tensor, output, average, name);                                        \
  }

#if !BYTEPS_GPU_PUSHPULL && HAVE_CUDA
PUSHPULL_CUDA_ON_CPU(torch_cuda_IntTensor, DataType::BYTEPS_INT32,
                      THCudaIntTensor, THIntTensor)
PUSHPULL_CUDA_ON_CPU(torch_cuda_LongTensor, DataType::BYTEPS_INT64,
                      THCudaLongTensor, THLongTensor)
PUSHPULL_CUDA_ON_CPU(torch_cuda_FloatTensor, DataType::BYTEPS_FLOAT32,
                      THCudaTensor, THFloatTensor)
PUSHPULL_CUDA_ON_CPU(torch_cuda_DoubleTensor, DataType::BYTEPS_FLOAT64,
                      THCudaDoubleTensor, THDoubleTensor)
#endif

extern "C" int byteps_torch_poll(int handle) {
  return handle_manager.PollHandle(handle) ? 1 : 0;
}

extern "C" void byteps_torch_wait_and_clear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

} // namespace torch
} // namespace byteps