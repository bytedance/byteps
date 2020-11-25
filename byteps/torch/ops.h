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

PUSHPULL_H(torch_ByteTensor, THByteTensor)
PUSHPULL_H(torch_IntTensor, THIntTensor)
PUSHPULL_H(torch_LongTensor, THLongTensor)
PUSHPULL_H(torch_FloatTensor, THFloatTensor)
PUSHPULL_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
PUSHPULL_H(torch_cuda_ByteTensor, THCudaByteTensor)
PUSHPULL_H(torch_cuda_IntTensor, THCudaIntTensor)
PUSHPULL_H(torch_cuda_LongTensor, THCudaLongTensor)
PUSHPULL_H(torch_cuda_FloatTensor, THCudaTensor)
PUSHPULL_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

extern "C" int byteps_torch_poll(int handle);
extern "C" void byteps_torch_wait_and_clear(int handle);
extern "C" void byteps_torch_declare_tensor(char* name);

}  // namespace torch
}  // namespace byteps

#endif  // BYTEPS_TORCH_OPS_H
