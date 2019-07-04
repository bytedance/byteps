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
    // cast src and dst to your float16 type
    auto* in = (unsigned short*)src;
    auto* inout = (unsigned short*)dst;

    int i = 0;
#if __AVX__ && __F16C__
    if (is_avx_and_f16c()) {
      for (; i < (int) (len / 8) * 8; i += 8) {
        // convert in & inout to m256
        __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in + i)));
        __m256 inout_m256 =
            _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(inout + i)));

        // add them together to new_inout_m256
        __m256 new_inout_m256 = _mm256_add_ps(in_m256, inout_m256);

        // convert back and store in inout
        __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
        _mm_storeu_si128((__m128i*)(inout + i), new_inout_m128i);
      }
    }
#endif
    for (; i < (int) len; ++i) {
      float in_float;
      float inout_float;
      HalfBits2Float(in + i, &in_float);
      HalfBits2Float(inout + i, &inout_float);
      inout_float += in_float;
      Float2HalfBits(&inout_float, inout + i);
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
    // cast src and dst to your float16 type
    auto* in1 = (unsigned short*)src1;
    auto* in2 = (unsigned short*)src2;
    auto* out = (unsigned short*)dst;

    int i = 0;
#if __AVX__ && __F16C__
    if (is_avx_and_f16c()) {
      for (; i < (int) (len / 8) * 8; i += 8) {
        // convert in1 & in2 to m256
        __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in1 + i)));
        __m256 inout_m256 =
            _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in2 + i)));

        // add them together to new_inout_m256
        __m256 new_inout_m256 = _mm256_add_ps(in_m256, inout_m256);

        // convert back and store in out
        __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
        _mm_storeu_si128((__m128i*)(out + i), new_inout_m128i);
      }
    }
#endif
    for (; i < (int) len; ++i) {
      float in1_float;
      float in2_float;
      float out_float;
      HalfBits2Float(in1 + i, &in1_float);
      HalfBits2Float(in2 + i, &in2_float);
      out_float = in1_float + in2_float;
      Float2HalfBits(&out_float, out + i);
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




} // namespace common
} // namespace byteps
