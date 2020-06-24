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

#include "util.h"

namespace byteps {
namespace sparse {


int createSharedMemory(const char *name, size_t sz, void** ptr) {
  int fd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (fd < 0) return errno;

  int status = ftruncate(fd, sz);
  if (status != 0) return status;

  void* addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == NULL) return errno;
  ptr = &addr;
  return 0;
}

int openSharedMemory(const char *name, size_t sz, void** ptr) {
  int fd = shm_open(name, O_RDWR, 0777);
  if (fd < 0) return errno;

  void* addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == NULL) return errno;
  ptr = &addr;
  return 0;
}

int createCudaIpcSharedMemory(const char *name, size_t sz, sharedMemoryInfo *info) {
  info->size = sz;
  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  int status = ftruncate(info->shmFd, sz);
  if (status != 0) {
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }
  return 0;
}


int openCudaIpcSharedMemory(const char *name, size_t sz, sharedMemoryInfo *info) {
  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0) {
    return errno;
  }
  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }
  return 0;
}


void closeCudaIpcSharedMemory(sharedMemoryInfo *info) {
  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shmFd) {
    close(info->shmFd);
  }
}

void mallocAligned(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  CHECK_EQ(ret, 0) 
      << "posix_memalign error: " << strerror(ret);
  CHECK(p);
  memset(p, 0, size);
  *ptr = p;
}

void mallocAlignedCudaAwareCpubuff(void **ptr, size_t size) {
  mallocAligned(ptr, size);
  CUDA_CALL(cudaHostRegister(*ptr, size, cudaHostRegisterMapped));
}

} // namespace sparse
} // namespace byteps