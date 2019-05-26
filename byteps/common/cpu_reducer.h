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

#ifndef BYTEPS_CPU_REDUCER_H
#define BYTEPS_CPU_REDUCER_H

#include <memory>
#include "common.h"
#include "communicator.h"

#define BYTEPS_CPU_REDUCER_THREADS 16

namespace byteps {
namespace common {

class CpuReducer {

public:
    CpuReducer(std::shared_ptr<BytePSComm> comm);
    int sum(void* dst, void* src, size_t len, DataType dtype);
    bool isRoot();
    std::shared_ptr<BytePSComm> getComm() { return _comm; }

private:
    int _sum_float32(void* dst, void* src, size_t len);
    int _sum_float64(void* dst, void* src, size_t len);
    int _sum_float16(void* dst, void* src, size_t len);
    int _sum_unit8(void* dst, void* src, size_t len);
    int _sum_int32(void* dst, void* src, size_t len);
    int _sum_int8(void* dst, void* src, size_t len);
    int _sum_int64(void* dst, void* src, size_t len);

    std::shared_ptr<BytePSComm> _comm;
    int _num_threads;
};


} // namespace common
} // namespace byteps

#endif // BYTEPS_CPU_REDUCER_H