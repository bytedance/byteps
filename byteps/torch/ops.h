// Copyright 2018 ByteDance, Inc. All Rights Reserved.
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

#ifndef BYTEPS_TORCH_OPS_H
#define BYTEPS_TORCH_OPS_H

#include <TH/TH.h>

#if HAVE_CUDA
#include <THC/THC.h>
#endif

#include "../common/operations.h"

namespace byteps {
namespace torch {

using namespace byteps::common;

std::mutex mutex_;
/* total number of gradients to push-pull */
size_t num_grads_;
/* number of push-pulls that have been triggered */
size_t grad_count_;

#define PUSHPULL_H(torch_Tensor, THTensor)                         \
  extern "C" int byteps_torch_push_pull_async_##torch_Tensor(      \
      THTensor* tensor, THTensor* output, int average, char* name, \
      int version, int priority);

#define CPU_COMPRESS_H(torch_Tensor, THTensor)                         \
  extern "C" int byteps_torch_cpu_compress_async_##torch_Tensor(      \
      THTensor* tensor, THTensor* output, int average, char* name, \
      int version, int priority);

#define INTRA_GATHER_H(torch_Tensor, THTensor)                         \
  extern "C" int byteps_torch_intra_gather_async_##torch_Tensor(      \
      THTensor* tensor, THTensor* output, int average, char* name, \
      int version, int priority, int root);

#define INTRA_BROADCAST_H(torch_Tensor, THTensor)                         \
  extern "C" int byteps_torch_intra_broadcast_async_##torch_Tensor(      \
      THTensor* tensor, THTensor* output, int average, char* name, \
      int version, int priority, int root);

#define INTRA_REDUCESCATTER_H(torch_Tensor, THTensor)                         \
  extern "C" int byteps_torch_intra_reducescatter_async_##torch_Tensor(      \
      THTensor* tensor, THTensor* output, int average, char* name, \
      int version, int priority);

#define INTRA_ALLGATHER_H(torch_Tensor, THTensor)                         \
  extern "C" int byteps_torch_intra_allgather_async_##torch_Tensor(      \
      THTensor* tensor, THTensor* output, int average, char* name, \
      int version, int priority);

#define INTRA_ALLTOALL_H(torch_Tensor, THTensor)                         \
  extern "C" int byteps_torch_intra_alltoall_async_##torch_Tensor(      \
      THTensor* tensor, THTensor* output, int average, char* name, \
      int version, int priority);

#define INTRA_REDUCE_H(torch_Tensor, THTensor)                         \
  extern "C" int byteps_torch_intra_reduce_async_##torch_Tensor(      \
      THTensor* tensor, THTensor* output, int average, char* name, \
      int version, int priority, int root);


PUSHPULL_H(torch_ByteTensor, THByteTensor)
PUSHPULL_H(torch_HalfTensor, THHalfTensor)
PUSHPULL_H(torch_IntTensor, THIntTensor)
PUSHPULL_H(torch_LongTensor, THLongTensor)
PUSHPULL_H(torch_FloatTensor, THFloatTensor)
PUSHPULL_H(torch_DoubleTensor, THDoubleTensor)

CPU_COMPRESS_H(torch_ByteTensor, THByteTensor)
CPU_COMPRESS_H(torch_HalfTensor, THHalfTensor)
CPU_COMPRESS_H(torch_IntTensor, THIntTensor)
CPU_COMPRESS_H(torch_LongTensor, THLongTensor)
CPU_COMPRESS_H(torch_FloatTensor, THFloatTensor)
CPU_COMPRESS_H(torch_DoubleTensor, THDoubleTensor)

INTRA_GATHER_H(torch_ByteTensor, THByteTensor)
INTRA_GATHER_H(torch_HalfTensor, THHalfTensor)
INTRA_GATHER_H(torch_IntTensor, THIntTensor)
INTRA_GATHER_H(torch_LongTensor, THLongTensor)
INTRA_GATHER_H(torch_FloatTensor, THFloatTensor)
INTRA_GATHER_H(torch_DoubleTensor, THDoubleTensor)

INTRA_BROADCAST_H(torch_ByteTensor, THByteTensor)
INTRA_BROADCAST_H(torch_HalfTensor, THHalfTensor)
INTRA_BROADCAST_H(torch_IntTensor, THIntTensor)
INTRA_BROADCAST_H(torch_LongTensor, THLongTensor)
INTRA_BROADCAST_H(torch_FloatTensor, THFloatTensor)
INTRA_BROADCAST_H(torch_DoubleTensor, THDoubleTensor)

INTRA_REDUCESCATTER_H(torch_ByteTensor, THByteTensor)
INTRA_REDUCESCATTER_H(torch_HalfTensor, THHalfTensor)
INTRA_REDUCESCATTER_H(torch_IntTensor, THIntTensor)
INTRA_REDUCESCATTER_H(torch_LongTensor, THLongTensor)
INTRA_REDUCESCATTER_H(torch_FloatTensor, THFloatTensor)
INTRA_REDUCESCATTER_H(torch_DoubleTensor, THDoubleTensor)

INTRA_ALLGATHER_H(torch_ByteTensor, THByteTensor)
INTRA_ALLGATHER_H(torch_HalfTensor, THHalfTensor)
INTRA_ALLGATHER_H(torch_IntTensor, THIntTensor)
INTRA_ALLGATHER_H(torch_LongTensor, THLongTensor)
INTRA_ALLGATHER_H(torch_FloatTensor, THFloatTensor)
INTRA_ALLGATHER_H(torch_DoubleTensor, THDoubleTensor)

INTRA_ALLTOALL_H(torch_ByteTensor, THByteTensor)
INTRA_ALLTOALL_H(torch_HalfTensor, THHalfTensor)
INTRA_ALLTOALL_H(torch_IntTensor, THIntTensor)
INTRA_ALLTOALL_H(torch_LongTensor, THLongTensor)
INTRA_ALLTOALL_H(torch_FloatTensor, THFloatTensor)
INTRA_ALLTOALL_H(torch_DoubleTensor, THDoubleTensor)

INTRA_REDUCE_H(torch_ByteTensor, THByteTensor)
INTRA_REDUCE_H(torch_HalfTensor, THHalfTensor)
INTRA_REDUCE_H(torch_IntTensor, THIntTensor)
INTRA_REDUCE_H(torch_LongTensor, THLongTensor)
INTRA_REDUCE_H(torch_FloatTensor, THFloatTensor)
INTRA_REDUCE_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
PUSHPULL_H(torch_cuda_ByteTensor, THCudaByteTensor)
PUSHPULL_H(torch_cuda_HalfTensor, THCudaHalfTensor)
PUSHPULL_H(torch_cuda_IntTensor, THCudaIntTensor)
PUSHPULL_H(torch_cuda_LongTensor, THCudaLongTensor)
PUSHPULL_H(torch_cuda_FloatTensor, THCudaTensor)
PUSHPULL_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)

CPU_COMPRESS_H(torch_cuda_ByteTensor, THCudaByteTensor)
CPU_COMPRESS_H(torch_cuda_HalfTensor, THCudaHalfTensor)
CPU_COMPRESS_H(torch_cuda_IntTensor, THCudaIntTensor)
CPU_COMPRESS_H(torch_cuda_LongTensor, THCudaLongTensor)
CPU_COMPRESS_H(torch_cuda_FloatTensor, THCudaTensor)
CPU_COMPRESS_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)

INTRA_GATHER_H(torch_cuda_ByteTensor, THCudaByteTensor)
INTRA_GATHER_H(torch_cuda_HalfTensor, THCudaHalfTensor)
INTRA_GATHER_H(torch_cuda_IntTensor, THCudaIntTensor)
INTRA_GATHER_H(torch_cuda_LongTensor, THCudaLongTensor)
INTRA_GATHER_H(torch_cuda_FloatTensor, THCudaTensor)
INTRA_GATHER_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)

INTRA_BROADCAST_H(torch_cuda_ByteTensor, THCudaByteTensor)
INTRA_BROADCAST_H(torch_cuda_HalfTensor, THCudaHalfTensor)
INTRA_BROADCAST_H(torch_cuda_IntTensor, THCudaIntTensor)
INTRA_BROADCAST_H(torch_cuda_LongTensor, THCudaLongTensor)
INTRA_BROADCAST_H(torch_cuda_FloatTensor, THCudaTensor)
INTRA_BROADCAST_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)

INTRA_REDUCESCATTER_H(torch_cuda_ByteTensor, THCudaByteTensor)
INTRA_REDUCESCATTER_H(torch_cuda_HalfTensor, THCudaHalfTensor)
INTRA_REDUCESCATTER_H(torch_cuda_IntTensor, THCudaIntTensor)
INTRA_REDUCESCATTER_H(torch_cuda_LongTensor, THCudaLongTensor)
INTRA_REDUCESCATTER_H(torch_cuda_FloatTensor, THCudaTensor)
INTRA_REDUCESCATTER_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)

INTRA_ALLGATHER_H(torch_cuda_ByteTensor, THCudaByteTensor)
INTRA_ALLGATHER_H(torch_cuda_HalfTensor, THCudaHalfTensor)
INTRA_ALLGATHER_H(torch_cuda_IntTensor, THCudaIntTensor)
INTRA_ALLGATHER_H(torch_cuda_LongTensor, THCudaLongTensor)
INTRA_ALLGATHER_H(torch_cuda_FloatTensor, THCudaTensor)
INTRA_ALLGATHER_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)

INTRA_ALLTOALL_H(torch_cuda_ByteTensor, THCudaByteTensor)
INTRA_ALLTOALL_H(torch_cuda_HalfTensor, THCudaHalfTensor)
INTRA_ALLTOALL_H(torch_cuda_IntTensor, THCudaIntTensor)
INTRA_ALLTOALL_H(torch_cuda_LongTensor, THCudaLongTensor)
INTRA_ALLTOALL_H(torch_cuda_FloatTensor, THCudaTensor)
INTRA_ALLTOALL_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)

INTRA_REDUCE_H(torch_cuda_ByteTensor, THCudaByteTensor)
INTRA_REDUCE_H(torch_cuda_HalfTensor, THCudaHalfTensor)
INTRA_REDUCE_H(torch_cuda_IntTensor, THCudaIntTensor)
INTRA_REDUCE_H(torch_cuda_LongTensor, THCudaLongTensor)
INTRA_REDUCE_H(torch_cuda_FloatTensor, THCudaTensor)
INTRA_REDUCE_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)

#endif

extern "C" int byteps_torch_poll(int handle);
extern "C" void byteps_torch_wait_and_clear(int handle);
extern "C" void byteps_torch_declare_tensor(char* name);
extern "C" void byteps_torch_declare_intra_reduce_tensor(char* name);
extern "C" void byteps_torch_declare_intra_gather_tensor(char* name);
extern "C" void byteps_torch_declare_intra_broadcast_tensor(char* name);
extern "C" void byteps_torch_declare_intra_reducescatter_tensor(char* name);
extern "C" void byteps_torch_declare_intra_allgather_tensor(char* name);
extern "C" void byteps_torch_declare_intra_alltoall_tensor(char* name);
extern "C" void byteps_torch_declare_cpu_compress_tensor(char* name, size_t size);


}  // namespace torch
}  // namespace byteps

#endif  // BYTEPS_TORCH_OPS_H
