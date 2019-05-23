// Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
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

#include <unistd.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>

#include "shared_memory.h"
#include "logging.h"

namespace byteps {
namespace common {

void* BytePSSharedMemory::openSharedMemory(int key, size_t size) {
    std::string shm_name("BytePS_ShM_");
    shm_name += std::to_string(key);
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    BPS_CHECK_GE(shm_fd, 0) << "shm_open failed for " << shm_name;

    BPS_CHECK_GE(ftruncate(shm_fd, size), 0) << strerror(errno);

    void* ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    BPS_CHECK_NE(ptr, (void *)-1) << strerror(errno);

    BPS_LOG(TRACE) << "initialized share memory size " << size;

    std::lock_guard<std::mutex> lock(_shm_mu);
    _key_shm_name[key] = shm_name;
    return ptr;
}

} // namespace common

} // namespace byteps