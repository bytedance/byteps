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

#ifndef BYTEPS_SPARSE_SERVER_H
#define BYTEPS_SPARSE_SERVER_H

#include <cuda_runtime.h>
#include <cstdlib>
#include <unistd.h>
#include "ps/ps.h"
#include "util.h"

namespace byteps {
namespace sparse {

#define DIVUP(x, y) (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) (DIVUP((x), (y))*(y))

extern "C" void bytepsSparseServer();

uint64_t DecodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key - kr.begin();
}

void MallocAligned(void** ptr, size_t size) {
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

template <typename T>
void AllocMemoryAndCreateSarray(ps::SArray<T>& sarr, T* addr, int count) {
  void* ptr;
  MallocAligned(&ptr, count * sizeof(T));
  memcpy(ptr, (void*)addr, count * sizeof(T));
  sarr.reset((T*)ptr, count, [](void *){});
}

} // namespace sparse
} // namespace byteps

#endif  // BYTEPS_SPARSE_SERVER_H
