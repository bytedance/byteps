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

#include "cpu_reducer.h"
#include "global.h"

namespace byteps {
namespace common {

CpuReducer::CpuReducer(std::shared_ptr<BytePSComm> comm) {
    std::vector<int> peers;
    auto pcie_size = BytePSGlobal::GetPcieSwitchSize();
    for (int i = BytePSGlobal::GetLocalRank() % pcie_size;
         i < BytePSGlobal::GetLocalSize();
         i += pcie_size) {
        peers.push_back(i);
    }
    _comm = std::make_shared<BytePSCommSocket>(comm, std::string("cpu"), peers);
    if (getenv("BYTEPS_OMP_THREAD_PER_GPU")) {
        _num_threads = atoi(getenv("BYTEPS_OMP_THREAD_PER_GPU"));
    }
    else {
        _num_threads = 4;
    }
    return;
}

bool CpuReducer::isRoot() {
    return (_comm->getRoot() == BytePSGlobal::GetLocalRank());
}

int CpuReducer::sum(void* dst, void* src, size_t len, DataType dtype) {
    switch (dtype) {
        case BYTEPS_FLOAT32:
            return _sum_float32(dst, src, len);
        case BYTEPS_FLOAT64:
            return _sum_float64(dst, src, len);
        case BYTEPS_FLOAT16:
            return _sum_float16(dst, src, len);
        case BYTEPS_UINT8:
            return _sum_unit8(dst, src, len);
        case BYTEPS_INT32:
            return _sum_int32(dst, src, len);
        case BYTEPS_INT8:
            return _sum_int8(dst, src, len);
        case BYTEPS_INT64:
            return _sum_int64(dst, src, len);
        default:
            BPS_CHECK(0) << "Unsupported data type: " << dtype;
    }
    return 0;
}

int CpuReducer::_sum_float32(void* dst, void* src, size_t len) {
    auto d = (float*)dst;
    auto s = (float*)src;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 4; ++i) {
        d[i] = d[i] + s[i];
    }
    return 0;
}

int CpuReducer::_sum_float64(void* dst, void* src, size_t len) {
    auto d = (double*)dst;
    auto s = (double*)src;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 8; ++i) {
        d[i] = d[i] + s[i];
    }
    return 0;
}

int CpuReducer::_sum_float16(void* dst, void* src, size_t len) {
    // convert half precision to fp32 --> do sum --> convert fp32 to half precision
    auto d = (uint16_t*) dst;
    auto s = (uint16_t*) src;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 2; ++i) {
        float d_fp32 = _convert_half_to_full_precision((uint16_t) d[i]);
        float s_fp32 = _convert_half_to_full_precision((uint16_t) s[i]);
        d_fp32 = d_fp32 + s_fp32;
        d[i] = _convert_full_to_half_precision((float) d_fp32);
    }
    return 0;
}

int CpuReducer::_sum_unit8(void* dst, void* src, size_t len) {
    auto d = (unsigned char*)dst;
    auto s = (unsigned char*)src;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len; ++i) {
        d[i] = d[i] + s[i];
    }
    return 0;
}

int CpuReducer::_sum_int32(void* dst, void* src, size_t len) {
    auto d = (int*)dst;
    auto s = (int*)src;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 4; ++i) {
        d[i] = d[i] + s[i];
    }
    return 0;
}

int CpuReducer::_sum_int8(void* dst, void* src, size_t len) {
    auto d = (signed char*)dst;
    auto s = (signed char*)src;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len; ++i) {
        d[i] = d[i] + s[i];
    }
    return 0;
}

int CpuReducer::_sum_int64(void* dst, void* src, size_t len) {
    auto d = (long long*)dst;
    auto s = (long long*)src;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 8; ++i) {
        d[i] = d[i] + s[i];
    }
    return 0;
}

int CpuReducer::sum(void* dst, void* src1, void* src2, size_t len, DataType dtype) {
    switch (dtype) {
        case BYTEPS_FLOAT32:
            return _sum_float32(dst, src1, src2, len);
        case BYTEPS_FLOAT64:
            return _sum_float64(dst, src1, src2, len);
        case BYTEPS_FLOAT16:
            return _sum_float16(dst, src1, src2, len);
        case BYTEPS_UINT8:
            return _sum_unit8(dst, src1, src2, len);
        case BYTEPS_INT32:
            return _sum_int32(dst, src1, src2, len);
        case BYTEPS_INT8:
            return _sum_int8(dst, src1, src2, len);
        case BYTEPS_INT64:
            return _sum_int64(dst, src1, src2, len);
        default:
            BPS_CHECK(0) << "Unsupported data type: " << dtype;
    }
    return 0;
}

int CpuReducer::_sum_float32(void* dst, void* src1, void* src2, size_t len) {
    auto d = (float*)dst;
    auto s1 = (float*)src1;
    auto s2 = (float*)src2;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 4; ++i) {
        d[i] = s1[i] + s2[i];
    }
    return 0;
}

int CpuReducer::_sum_float64(void* dst, void* src1, void* src2, size_t len) {
    auto d = (double*)dst;
    auto s1 = (double*)src1;
    auto s2 = (double*)src2;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 8; ++i) {
        d[i] = s1[i] + s2[i];
    }
    return 0;
}

int CpuReducer::_sum_float16(void* dst, void* src1, void* src2, size_t len) {
    // convert half precision to fp32 --> do sum --> convert fp32 to half precision
    auto d = (uint16_t*) dst;
    auto s1 = (uint16_t*) src1;
    auto s2 = (uint16_t*) src2;

#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 2; ++i) {
        float d_fp32 = _convert_half_to_full_precision((uint16_t) d[i]);
        float s1_fp32 = _convert_half_to_full_precision((uint16_t) s1[i]);
        float s2_fp32 = _convert_half_to_full_precision((uint16_t) s2[i]);
        d_fp32 = s1_fp32 + s2_fp32;
        d[i] = _convert_full_to_half_precision((float) d_fp32);
    }
    return 0;
}

int CpuReducer::_sum_unit8(void* dst, void* src1, void* src2, size_t len) {
    auto d = (unsigned char*)dst;
    auto s1 = (unsigned char*)src1;
    auto s2 = (unsigned char*)src2;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len; ++i) {
        d[i] = s1[i] + s2[i];
    }
    return 0;
}

int CpuReducer::_sum_int32(void* dst, void* src1, void* src2, size_t len) {
    auto d = (int*)dst;
    auto s1 = (int*)src1;
    auto s2 = (int*)src2;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 4; ++i) {
        d[i] = s1[i] + s2[i];
    }
    return 0;
}

int CpuReducer::_sum_int8(void* dst, void* src1, void* src2, size_t len) {
    auto d = (signed char*)dst;
    auto s1 = (signed char*)src1;
    auto s2 = (signed char*)src2;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len; ++i) {
        d[i] = s1[i] + s2[i];
    }
    return 0;
}

int CpuReducer::_sum_int64(void* dst, void* src1, void* src2, size_t len) {
    auto d = (long long*)dst;
    auto s1 = (long long*)src1;
    auto s2 = (long long*)src2;
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < len / (size_t) 8; ++i) {
        d[i] = s1[i] + s2[i];
    }
    return 0;
}

float CpuReducer::_convert_half_to_full_precision(uint16_t h) {
    float f = ((h&0x8000)<<16) | (((h&0x7c00)+0x1C000)<<13) | ((h&0x03FF)<<13);
    return f;
}

uint16_t CpuReducer::_convert_full_to_half_precision(float f) {
    uint32_t x = *((uint32_t*)&f);
    uint16_t h = ((x>>16)&0x8000) | ((((x&0x7f800000)-0x38000000)>>13)&0x7c00) | ((x>>13)&0x03ff);
    return h;
}


} // namespace common
} // namespace byteps
