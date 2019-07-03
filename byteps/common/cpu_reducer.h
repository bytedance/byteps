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

#ifndef BYTEPS_CPU_REDUCER_H
#define BYTEPS_CPU_REDUCER_H

#include <memory>
#include "common.h"
#include "communicator.h"
#include "logging.h"

namespace byteps {
namespace common {

class CpuReducer {

public:
    CpuReducer(std::shared_ptr<BytePSComm> comm);
    ~CpuReducer() {
        if (_comm) _comm.reset();
        BPS_LOG(DEBUG) << "Clear CpuReducer";
    }
    
    int sum(void* dst, void* src, size_t len, DataType dtype);
    int sum(void* dst, void* src1, void* src2, size_t len, DataType dtype);
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

    int _sum_float32(void* dst, void* src1, void* src2, size_t len);
    int _sum_float64(void* dst, void* src1, void* src2, size_t len);
    int _sum_float16(void* dst, void* src1, void* src2, size_t len);
    int _sum_unit8(void* dst, void* src1, void* src2, size_t len);
    int _sum_int32(void* dst, void* src1, void* src2, size_t len);
    int _sum_int8(void* dst, void* src1, void* src2, size_t len);
    int _sum_int64(void* dst, void* src1, void* src2, size_t len);

    float _convert_half_to_full_precision(uint16_t h);
    uint16_t _convert_full_to_half_precision(float f);

    std::shared_ptr<BytePSComm> _comm;
    int _num_threads;
};


} // namespace common
} // namespace byteps

#endif // BYTEPS_CPU_REDUCER_H