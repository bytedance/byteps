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

    std::string shm_name("BytePS_");
    shm_name += std::to_string(key);
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, size);
    void* ptr = mmap(0, size, PROT_WRITE, MAP_SHARED, shm_fd, 0); 
    std::lock_guard<std::mutex> lock(_shm_mu);
    _key_shm_addr[key] = ptr;
    _key_shm_len[key] = size;
    return ptr;
}

void* BytePSSharedMemory::getSharedMemoryAddress(int key) {
    std::lock_guard<std::mutex> lock(_shm_mu);
    return _key_shm_addr[key];
}

int BytePSSharedMemory::getSharedMemoryLen(int key) {
    std::lock_guard<std::mutex> lock(_shm_mu);
    return _key_shm_len[key];
}



} // namespace common

} // namespace byteps