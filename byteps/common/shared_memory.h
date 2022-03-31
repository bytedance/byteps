// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_SHARED_MEMORY_H
#define BYTEPS_SHARED_MEMORY_H

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "logging.h"

namespace byteps {
namespace common {

class BytePSSharedMemory {
 public:
  BytePSSharedMemory() {}

  ~BytePSSharedMemory() {
    for (auto &it : _key_shm_addr) {
      CUDA_CALL(cudaHostUnregister(it.second));
      munmap(it.second, _key_shm_size[it.first]);
      shm_unlink(it.first.c_str());
    }

    BPS_LOG(DEBUG) << "Clear shared memory: all BytePS shared memory "
                      "released/unregistered.";
  }

  void *openSharedMemory(const std::string &prefix, uint64_t key, size_t size);
  std::vector<void *> openPcieSharedMemory(uint64_t key, size_t size);

 private:
  std::unordered_map<std::string, void *> _key_shm_addr;
  std::unordered_map<std::string, size_t> _key_shm_size;

  std::mutex _shm_mu;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SHARED_MEMORY_H
