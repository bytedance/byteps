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
                                           uint64_t key, size_t len) {
  std::string shm_name(prefix);
  shm_name += std::to_string(key);
  auto size = BytePSGlobal::RoundUpToPageSize(len);
  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  BPS_CHECK_GE(shm_fd, 0) << "shm_open failed for " << shm_name;

  BPS_CHECK_GE(ftruncate(shm_fd, size), 0) << strerror(errno);

  void* ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  CHECK_NE(ptr, (void*)-1) << strerror(errno)
    << "mmap failed. prefix = " << prefix
    << ", key = " << key << ", size = " << size;

#if BYTEPS_BUILDING_CUDA == 1
  {
    cudaError_t e = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    BPS_CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)
      << "CUDA: " << cudaGetErrorString(e)
      << ". Invalid pointer. prefix = " << prefix
      << ", key = " << key << ", size = " << size
      << ", error = " << e << (e == cudaErrorInvalidValue ?
      ". Please check if you have enough space for shared memory (df -h /dev/shm)" : ".");
  }
  // mlock(ptr, size);
#endif

  BPS_CHECK_NE(ptr, (void*)-1) << strerror(errno);
  BPS_LOG(DEBUG) << "initialized share memory size " << size << ", name=" << shm_name
                 << ", len=" << len;

  std::lock_guard<std::mutex> lock(_shm_mu);
  _key_shm_addr[shm_name] = ptr;
  _key_shm_size[shm_name] = size;

  return ptr;
}

std::vector<void*> BytePSSharedMemory::openPcieSharedMemory(const std::string& prefix,
                                                            uint64_t key,
                                                            size_t size) {
  std::vector<void*> r;
#if BYTEPS_BUILDING_CUDA == 1
  for (int i = 0; i < BytePSGlobal::GetPcieSwitchNum(); i++) {
    auto prefix_i = prefix + std::to_string(i) + "_Shm_";
    if (BytePSGlobal::IsDistributed()) {
      if (BytePSGlobal::IsCrossPcieSwitch()) {
        if (i <= numa_max_node()) {
          numa_set_preferred(i);
          r.push_back(openSharedMemory(prefix_i, key, size));
          numa_set_preferred(-1);
        } else {
          numa_set_preferred(numa_max_node());
          r.push_back(openSharedMemory(prefix_i, key, size));
          numa_set_preferred(-1);
        }
      } else {
        r.push_back(openSharedMemory(prefix_i, key, size));
      }
    } else {
      if (BytePSGlobal::IsCrossPcieSwitch()) {
        numa_set_interleave_mask(numa_all_nodes_ptr);
        r.push_back(openSharedMemory(prefix_i, key, size));
        numa_set_interleave_mask(numa_no_nodes_ptr);
      } else {
        r.push_back(openSharedMemory(prefix_i, key, size));
      }
    }
  }
#else
  BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
  return r;
}

std::vector<void*> BytePSSharedMemory::openNumaSharedMemory(const std::string& prefix,
                                                            uint64_t key,
                                                            size_t size) {
  std::vector<void*> ret;

  for (int i = 0; i < BytePSGlobal::GetLocalSize(); i++) {
    std::string prefix_i = prefix + std::to_string(BytePSGlobal::GetPhyNodeID())
                         + std::string("_") + std::to_string(i) + "_ShM_";
    if (i <= numa_max_node()) {
      numa_set_preferred(i);
    } else {
      numa_set_preferred(numa_max_node());
    }
    ret.push_back(openSharedMemory(prefix_i, key, size));
    numa_set_preferred(-1);
  }

  return ret;
}


}  // namespace common

}  // namespace byteps
