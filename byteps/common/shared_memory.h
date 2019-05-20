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

#ifndef BYTEPS_SHARED_MEMORY_H
#define BYTEPS_SHARED_MEMORY_H

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <cerrno>

#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>


#define BYTEPS_SHAREDMEMORY_BASENAME "BYTEPS_SHARED_MEMORY"


namespace byteps {
namespace common {

class BytePSSharedMemory {

public:

    BytePSSharedMemory() {}

    ~BytePSSharedMemory() {
        //shm_unlink(name);

    }

    void* produceSharedMemory(int key);

    void* getSharedMemoryAddress(int destination, int key);

    int getSharedMemoryLen(int destination, int key);

    bool isSharedMemoryExist(int destination, int key);


private:

    int _shm_fd;

    int _root_rank;
    int _my_rank;
    std::unordered_map<int, void*> _key_shm_addr;
    std::unordered_map<int, int> _key_shm_len;

    std::mutex _shm_mu;

};


} // namespace common
} // namespace byteps

#endif // BYTEPS_SHARED_MEMORY_H