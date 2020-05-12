// Copyright 2020 Bytedance Inc. or its affiliates. All Rights Reserved.
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


#ifndef BYTEPS_SPARSE_H
#define BYTEPS_SPARSE_H

#include <vector>
#include <cuda_runtime.h>
#include "ops.h"
#include "adapter.h"
#include "comm.h"
#include "ps/ps.h"

namespace byteps {
namespace sparse {

#define MAX_CUDA_DEVICES (32)

static std::vector<void*> _embedBuffers;
static std::vector<void*> _denseBuffers;
static ps::KVWorker<char>* _ps;
static void* _cpuBuffer;


typedef struct shmStruct_st {
  size_t nprocesses;
  int devices[MAX_CUDA_DEVICES];
  cudaIpcMemHandle_t embedMemHandle[MAX_CUDA_DEVICES];
  cudaIpcMemHandle_t denseMemHandle[MAX_CUDA_DEVICES];
  cudaIpcEventHandle_t eventHandle[MAX_CUDA_DEVICES];
} shmStruct;

extern "C" void bytepsSparseInit(std::vector<void*>& embedBuffers, std::vector<void*>& denseBuffers, int size);

extern "C" void bytepsSparseShutdown();

extern "C" void bytepsGather(int rank, int len, cudaStream_t stream);

extern "C" void bytepsScatter(int rank, int len, cudaStream_t stream);

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_H
