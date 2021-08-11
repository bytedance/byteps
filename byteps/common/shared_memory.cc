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

#include "shared_memory.h"
#include <fcntl.h>
#include <numa.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "global.h"

namespace byteps {
namespace common {

void* BytePSSharedMemory::openSharedMemory(const std::string& prefix,
                                           uint64_t key, size_t size) {
  size = BytePSGlobal::RoundUpToPageSize(size);
  std::string shm_name(prefix);
  shm_name += std::to_string(key);
  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  BPS_CHECK_GE(shm_fd, 0) << "shm_open failed for " << shm_name << " " << strerror(errno);

  BPS_CHECK_GE(ftruncate(shm_fd, size), 0) << strerror(errno);

  void* ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  CUDA_CALL(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
  // mlock(ptr, size);

  BPS_CHECK_NE(ptr, (void*)-1) << strerror(errno);

  BPS_LOG(TRACE) << "initialized share memory size " << size;

  std::lock_guard<std::mutex> lock(_shm_mu);
  _key_shm_addr[shm_name] = ptr;
  _key_shm_size[shm_name] = size;
  return ptr;
}

std::vector<void*> BytePSSharedMemory::openPcieSharedMemory(uint64_t key,
                                                            size_t size) {
  std::vector<void*> r;
  for (int i = 0; i < BytePSGlobal::GetPcieSwitchNum(); i++) {
    auto prefix = std::string("BytePS_Pcie") + std::to_string(i) + "_Shm_";
    if (BytePSGlobal::IsDistributed()) {
      if (BytePSGlobal::IsCrossPcieSwitch()) {
        if (i <= numa_max_node()) {
          numa_set_preferred(i);
          r.push_back(openSharedMemory(prefix, key, size));
          numa_set_preferred(-1);
        } else {
          numa_set_preferred(numa_max_node());
          r.push_back(openSharedMemory(prefix, key, size));
          numa_set_preferred(-1);
        }
      } else {
        r.push_back(openSharedMemory(prefix, key, size));
      }
    } else {
      if (BytePSGlobal::IsCrossPcieSwitch()) {
        numa_set_interleave_mask(numa_all_nodes_ptr);
        r.push_back(openSharedMemory(prefix, key, size));
        numa_set_interleave_mask(numa_no_nodes_ptr);
      } else {
        r.push_back(openSharedMemory(prefix, key, size));
      }
    }
  }
  return r;
}

}  // namespace common

}  // namespace byteps
