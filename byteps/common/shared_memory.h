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
#include <string.h>
#define BYTEPS_SHAREDMEMORY_BASENAME "BYTEPS_SHARED_MEMORY"


namespace byteps {
namespace common {

class BytePSSharedMemory {

public:

    BytePSSharedMemory() {}

    ~BytePSSharedMemory() {
        for (auto &it : _key_shm_name) {
            shm_unlink(it.second.c_str());
        }
    }

    void* openSharedMemory(int key, size_t size);

private:

    std::unordered_map<int, std::string> _key_shm_name;

    std::mutex _shm_mu;

};


} // namespace common
} // namespace byteps

#endif // BYTEPS_SHARED_MEMORY_H