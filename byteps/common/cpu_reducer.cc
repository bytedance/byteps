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

#ifndef BYTEPS_BUILDING_SERVER
#include "global.h"
#endif

#include <cmath>

#include "cpu_reducer.h"

namespace byteps {
namespace common {

CpuReducer::CpuReducer(std::shared_ptr<BytePSComm> comm) {
#ifndef BYTEPS_BUILDING_SERVER
  std::vector<int> peers;
  auto pcie_size = BytePSGlobal::GetPcieSwitchSize();
  for (int i = BytePSGlobal::GetLocalRank() % pcie_size;
       i < BytePSGlobal::GetLocalSize(); i += pcie_size) {
    peers.push_back(i);
  }
  if (comm) {
    _comm = std::make_shared<BytePSCommSocket>(comm, std::string("cpu"), peers);
  } else {
    _comm = nullptr;
  }
#endif
  if (getenv("BYTEPS_OMP_THREAD_PER_GPU")) {
    _num_threads = atoi(getenv("BYTEPS_OMP_THREAD_PER_GPU"));
  } else {
    _num_threads = 4;
  }

  return;
}

#ifndef BYTEPS_BUILDING_SERVER
bool CpuReducer::isRoot() {
  if (!_comm) {
    return false;
  }
  return (_comm->getRoot() == BytePSGlobal::GetLocalRank());
}
#endif

int CpuReducer::sum(void* dst, const void* src, size_t len, DataType dtype) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return _sum(reinterpret_cast<float*>(dst),
                  reinterpret_cast<const float*>(src), len);
    case BYTEPS_FLOAT64:
      return _sum(reinterpret_cast<double*>(dst),
                  reinterpret_cast<const double*>(src), len);
    case BYTEPS_FLOAT16:
      return _sum_float16(dst, src, len);
    case BYTEPS_UINT8:
      return _sum(reinterpret_cast<uint8_t*>(dst),
                  reinterpret_cast<const uint8_t*>(src), len);
    case BYTEPS_INT32:
      return _sum(reinterpret_cast<int32_t*>(dst),
                  reinterpret_cast<const int32_t*>(src), len);
    case BYTEPS_INT8:
      return _sum(reinterpret_cast<int8_t*>(dst),
                  reinterpret_cast<const int8_t*>(src), len);
    case BYTEPS_INT64:
      return _sum(reinterpret_cast<int64_t*>(dst),
                  reinterpret_cast<const int64_t*>(src), len);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

template <typename T>
int CpuReducer::_sum(T* dst, const T* src, size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] = dst[i] + src[i];
  }
  return 0;
}

int CpuReducer::_sum_float16(void* dst, const void* src, size_t len) {
  // cast src and dst to your float16 type
  auto in = reinterpret_cast<const unsigned short*>(src);
  auto inout = reinterpret_cast<unsigned short*>(dst);
  len = len / (size_t)2;

#if __AVX__ && __F16C__
  if (is_avx_and_f16c()) {
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < (size_t)(len / 8) * 8; i += 8) {
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

  for (size_t i = (len / 8) * 8; i < (size_t)len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float += in_float;
    Float2HalfBits(&inout_float, inout + i);
  }
#else
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < (size_t)len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float += in_float;
    Float2HalfBits(&inout_float, inout + i);
  }
#endif

  return 0;
}

int CpuReducer::sum(void* dst, const void* src1, const void* src2, size_t len,
                    DataType dtype) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return _sum(reinterpret_cast<float*>(dst),
                  reinterpret_cast<const float*>(src1),
                  reinterpret_cast<const float*>(src2), len);
    case BYTEPS_FLOAT64:
      return _sum(reinterpret_cast<double*>(dst),
                  reinterpret_cast<const double*>(src1),
                  reinterpret_cast<const double*>(src2), len);
    case BYTEPS_FLOAT16:
      return _sum_float16(dst, src1, src2, len);
    case BYTEPS_UINT8:
      return _sum(reinterpret_cast<uint8_t*>(dst),
                  reinterpret_cast<const uint8_t*>(src1),
                  reinterpret_cast<const uint8_t*>(src2), len);
    case BYTEPS_INT32:
      return _sum(reinterpret_cast<int32_t*>(dst),
                  reinterpret_cast<const int32_t*>(src1),
                  reinterpret_cast<const int32_t*>(src2), len);
    case BYTEPS_INT8:
      return _sum(reinterpret_cast<int8_t*>(dst),
                  reinterpret_cast<const int8_t*>(src1),
                  reinterpret_cast<const int8_t*>(src2), len);
    case BYTEPS_INT64:
      return _sum(reinterpret_cast<int64_t*>(dst),
                  reinterpret_cast<const int64_t*>(src1),
                  reinterpret_cast<const int64_t*>(src2), len);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

template <typename T>
int CpuReducer::_sum(T* dst, const T* src1, const T* src2, size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] = src1[i] + src2[i];
  }
  return 0;
}

int CpuReducer::_sum_float16(void* dst, const void* src1, const void* src2,
                             size_t len) {
  // cast src and dst to your float16 type
  auto in1 = reinterpret_cast<const unsigned short*>(src1);
  auto in2 = reinterpret_cast<const unsigned short*>(src2);
  auto out = reinterpret_cast<unsigned short*>(dst);
  len = len / (size_t)2;

#if __AVX__ && __F16C__
  if (is_avx_and_f16c()) {
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < (size_t)(len / 8) * 8; i += 8) {
      // convert in1 & in2 to m256
      __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in1 + i)));
      __m256 inout_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in2 + i)));

      // add them together to new_inout_m256
      __m256 new_inout_m256 = _mm256_add_ps(in_m256, inout_m256);

      // convert back and store in out
      __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
      _mm_storeu_si128((__m128i*)(out + i), new_inout_m128i);
    }
  }

  for (size_t i = (size_t)(len / 8) * 8; i < (size_t)len; ++i) {
    float in1_float;
    float in2_float;
    float out_float;
    HalfBits2Float(in1 + i, &in1_float);
    HalfBits2Float(in2 + i, &in2_float);
    out_float = in1_float + in2_float;
    Float2HalfBits(&out_float, out + i);
  }
#else
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < (size_t)len; ++i) {
    float in1_float;
    float in2_float;
    float out_float;
    HalfBits2Float(in1 + i, &in1_float);
    HalfBits2Float(in2 + i, &in2_float);
    out_float = in1_float + in2_float;
    Float2HalfBits(&out_float, out + i);
  }
#endif
  return 0;
}

int CpuReducer::sum(void* dst, const void* src, size_t len, DataType dtype,
                    float alpha) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return _sum(reinterpret_cast<float*>(dst),
                  reinterpret_cast<const float*>(src), len, alpha);
    case BYTEPS_FLOAT64:
      return _sum(reinterpret_cast<double*>(dst),
                  reinterpret_cast<const double*>(src), len, alpha);
    case BYTEPS_FLOAT16:
      return _sum_float16(dst, src, len, alpha);
    case BYTEPS_UINT8:
      return _sum(reinterpret_cast<uint8_t*>(dst),
                  reinterpret_cast<const uint8_t*>(src), len, alpha);
    case BYTEPS_INT32:
      return _sum(reinterpret_cast<int32_t*>(dst),
                  reinterpret_cast<const int32_t*>(src), len, alpha);
    case BYTEPS_INT8:
      return _sum(reinterpret_cast<int8_t*>(dst),
                  reinterpret_cast<const int8_t*>(src), len, alpha);
    case BYTEPS_INT64:
      return _sum(reinterpret_cast<int64_t*>(dst),
                  reinterpret_cast<const int64_t*>(src), len, alpha);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

template <typename T>
int CpuReducer::_sum(T* dst, const T* src, size_t len, float alpha) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] = dst[i] + alpha * src[i];
  }
  return 0;
}

int CpuReducer::_sum_float16(void* dst, const void* src, size_t len,
                             float alpha) {
  // cast src and dst to your float16 type
  auto in = reinterpret_cast<const unsigned short*>(src);
  auto inout = reinterpret_cast<unsigned short*>(dst);
  len = len / (size_t)2;

#if __AVX__ && __F16C__
  float mm256_alpha[8];
  for (int i = 0; i < 8; ++i) mm256_alpha[i] = alpha;

  if (is_avx_and_f16c()) {
    __m256 __mm256_alpha = _mm256_loadu_ps(mm256_alpha);
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < (size_t)(len / 8) * 8; i += 8) {
      // convert in & inout to m256
      __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in + i)));
      __m256 inout_m256 =
          _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(inout + i)));

      __m256 scaled_in_m256 = _mm256_mul_ps(in_m256, __mm256_alpha);
      // add them together to new_inout_m256
      __m256 new_inout_m256 = _mm256_add_ps(scaled_in_m256, inout_m256);

      // convert back and store in inout
      __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
      _mm_storeu_si128((__m128i*)(inout + i), new_inout_m128i);
    }
  }

  for (size_t i = (len / 8) * 8; i < (size_t)len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float += in_float * alpha;
    Float2HalfBits(&inout_float, inout + i);
  }
#else
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < (size_t)len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float += in_float * alpha;
    Float2HalfBits(&inout_float, inout + i);
  }
#endif

  return 0;
}

int CpuReducer::sum(void* dst, const void* src1, const void* src2, size_t len,
                    DataType dtype, float alpha) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return _sum(reinterpret_cast<float*>(dst),
                  reinterpret_cast<const float*>(src1),
                  reinterpret_cast<const float*>(src2), len, alpha);
    case BYTEPS_FLOAT64:
      return _sum(reinterpret_cast<double*>(dst),
                  reinterpret_cast<const double*>(src1),
                  reinterpret_cast<const double*>(src2), len, alpha);
    case BYTEPS_FLOAT16:
      return _sum_float16(dst, src1, src2, len, alpha);
    case BYTEPS_UINT8:
      return _sum(reinterpret_cast<uint8_t*>(dst),
                  reinterpret_cast<const uint8_t*>(src1),
                  reinterpret_cast<const uint8_t*>(src2), len, alpha);
    case BYTEPS_INT32:
      return _sum(reinterpret_cast<int32_t*>(dst),
                  reinterpret_cast<const int32_t*>(src1),
                  reinterpret_cast<const int32_t*>(src2), len, alpha);
    case BYTEPS_INT8:
      return _sum(reinterpret_cast<int8_t*>(dst),
                  reinterpret_cast<const int8_t*>(src1),
                  reinterpret_cast<const int8_t*>(src2), len, alpha);
    case BYTEPS_INT64:
      return _sum(reinterpret_cast<int64_t*>(dst),
                  reinterpret_cast<const int64_t*>(src1),
                  reinterpret_cast<const int64_t*>(src2), len, alpha);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

template <typename T>
int CpuReducer::_sum(T* dst, const T* src1, const T* src2, size_t len,
                     float alpha) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] = src1[i] + alpha * src2[i];
  }
  return 0;
}

int CpuReducer::_sum_float16(void* dst, const void* src1, const void* src2,
                             size_t len, float alpha) {
  // cast src and dst to your float16 type
  auto in1 = reinterpret_cast<const unsigned short*>(src1);
  auto in2 = reinterpret_cast<const unsigned short*>(src2);
  auto out = reinterpret_cast<unsigned short*>(dst);
  len = len / (size_t)2;

#if __AVX__ && __F16C__
  float mm256_alpha[8];
  for (int i = 0; i < 8; ++i) mm256_alpha[i] = alpha;

  if (is_avx_and_f16c()) {
    __m256 __mm256_alpha = _mm256_loadu_ps(mm256_alpha);
#pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < (size_t)(len / 8) * 8; i += 8) {
      // convert in1 & in2 to m256
      __m256 in1_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in1 + i)));
      __m256 in2_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in2 + i)));

      __m256 scaled_in2_m256 = _mm256_mul_ps(in2_m256, __mm256_alpha);
      // add them together to new_inout_m256
      __m256 new_out_m256 = _mm256_add_ps(in1_m256, scaled_in2_m256);

      // convert back and store in out
      __m128i new_out_m128i = _mm256_cvtps_ph(new_out_m256, 0);
      _mm_storeu_si128((__m128i*)(out + i), new_out_m128i);
    }
  }

  for (size_t i = (size_t)(len / 8) * 8; i < (size_t)len; ++i) {
    float in1_float;
    float in2_float;
    float out_float;
    HalfBits2Float(in1 + i, &in1_float);
    HalfBits2Float(in2 + i, &in2_float);
    out_float = in1_float + in2_float * alpha;
    Float2HalfBits(&out_float, out + i);
  }
#else
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < (size_t)len; ++i) {
    float in1_float;
    float in2_float;
    float out_float;
    HalfBits2Float(in1 + i, &in1_float);
    HalfBits2Float(in2 + i, &in2_float);
    out_float = in1_float + in2_float * alpha;
    Float2HalfBits(&out_float, out + i);
  }
#endif
  return 0;
}

int CpuReducer::copy(void* dst, const void* src, size_t len) {
  auto in = (float*)src;
  auto out = (float*)dst;
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / 4; ++i) {
    out[i] = in[i];
  }
  if (len % 4) {
    std::memcpy(out + len / 4, in + len / 4, len % 4);
  }
  return 0;
}
}  // namespace common
}  // namespace byteps
