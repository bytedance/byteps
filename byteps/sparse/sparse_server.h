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

extern "C" void bytepsSparseServer();

uint64_t decodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key - kr.begin();
}

template <typename T>
void allocMemoryAndCreateSarray(ps::SArray<T>& sarr, T* addr, int count) {
  void* ptr;
  mallocAligned(&ptr, count * sizeof(T));
  memcpy(ptr, (void*)addr, count * sizeof(T));
  sarr.reset((T*)ptr, count, [](void *){});
}

} // namespace sparse
} // namespace byteps

#endif  // BYTEPS_SPARSE_SERVER_H
