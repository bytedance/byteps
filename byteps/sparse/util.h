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

#ifndef BYTEPS_SPARSE_UTIL_H
#define BYTEPS_SPARSE_UTIL_H

#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <ps/ps.h>

namespace byteps {
namespace sparse {

#define MAX_CUDA_DEVICES (32)

#define CUDA_CALL(func)                                          \
  {                                                              \
    cudaError_t e = (func);                                      \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                    \
  }

static const char* bpsShmName = "BytePS_Sparse_ShM_";

typedef struct sharedMemoryInfo_st {
  void *addr;
  size_t size;
  int shmFd;
} sharedMemoryInfo;

typedef struct shmStruct_st {
  size_t nprocesses;
  int devices[MAX_CUDA_DEVICES];
  cudaIpcMemHandle_t embedMemHandle[MAX_CUDA_DEVICES];
  cudaIpcMemHandle_t denseMemHandle[MAX_CUDA_DEVICES];
  cudaIpcEventHandle_t eventHandle[MAX_CUDA_DEVICES];
} shmStruct;


int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);

void sharedMemoryClose(sharedMemoryInfo *info);

} // namespace sparse
} // namespace byteps

#endif // BYTEPS_SPARSE_UTIL_H