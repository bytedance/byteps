// Copyright 2019 Amazon Inc. or its affiliates. All Rights Reserved.
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

#if __AVX__ && __F16C__
#include <cpuid.h>
#include <immintrin.h>
#endif
#include <cstring>
#include "compressor.h"

namespace byteps {
namespace common {
namespace compressor {

namespace {
#if __AVX__ && __F16C__
  // Query CPUID to determine AVX and F16C runtime support.
  bool is_avx_and_f16c() {
    static bool initialized = false;
    static bool result = false;
    if (!initialized) {
      unsigned int eax, ebx, ecx, edx;
      if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        result = (ecx & bit_AVX) && (ecx & bit_F16C);
      }
      initialized = true;
    }
    return result;
  }
#endif

  inline void HalfBits2Float(const unsigned short* src, float* res) {
    unsigned h = *src;
    int sign = ((h >> 15) & 1);
    int exp = ((h >> 10) & 0x1f);
    int mantissa = (h & 0x3ff);
    unsigned f = 0;

    if (exp > 0 && exp < 31) {
      // normal
      exp += 112;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    } else if (exp == 0) {
      if (mantissa) {
        // subnormal
        exp += 113;
        while ((mantissa & (1 << 10)) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x3ff;
        f = (sign << 31) | (exp << 23) | (mantissa << 13);
      } else {
        // sign-preserving zero
        f = (sign << 31);
      }
    } else if (exp == 31) {
      if (mantissa) {
        f = 0x7fffffff;  // not a number
      } else {
        f = (0xff << 23) | (sign << 31);  //  inf
      }
    }

    *res = *reinterpret_cast<float const*>(&f);
  }

  inline void Float2HalfBits(const float* src, unsigned short* dest) {
    // software implementation rounds toward nearest even
    unsigned const& s = *reinterpret_cast<unsigned const*>(src);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      *dest = sign;
      return;
    }

    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7c00;
      }
      *dest = u;
      return;
    }

    int sticky_bit = 0;

    if (exp >= -14) {
      // normal fp32 to normal fp16
      exp = uint16_t(exp + uint16_t(15));
      u = uint16_t(((exp & 0x1f) << 10));
      u = uint16_t(u | (mantissa >> 13));
    } else {
      // normal single-precision to subnormal half_t-precision representation
      int rshift = (-14 - exp);
      if (rshift < 32) {
        mantissa |= (1 << 23);

        sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 13) & 0x3ff);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    // round to nearest even
    int round_bit = ((mantissa >> 12) & 1);
    sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
      u = uint16_t(u + 1);
    }

    u |= sign;

    *dest = u;
  }
}


  void Compressor::FP16toFP32(void* dst, void* src, size_t len) {
    auto in = reinterpret_cast<const unsigned short*>(src);
    auto out = reinterpret_cast<float*>(dst);
    len = len / (size_t)2;

  #if __AVX__ && __F16C__
    if (is_avx_and_f16c()) {
  #pragma omp parallel for simd num_threads(_num_threads)
      for (size_t i = 0; i < (size_t)(len / 8) * 8; i += 8) {
        // convert to m256
        _mm256_storeu_si256((__m256i*)(out+i), (__m256i)_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in + i))));
      }
    }

    for (size_t i = (len / 8) * 8; i < (size_t)len; ++i) {
      float in_float;
      HalfBits2Float(in + i, &in_float);
      out[i] = in_float;
    }
  #else
  #pragma omp parallel for simd num_threads(_num_threads)
    for (size_t i = 0; i < (size_t)len; ++i) {
      HalfBits2Float(in + i, &(out + i));
    }
  #endif
  }

  void Compressor::FP32toFP16(void* dst, void* src, size_t len) {
    auto in = reinterpret_cast<const float*>(src);
    auto out = reinterpret_cast<unsigned short*>(dst);
    len = len / (size_t)4;
    
  #if __AVX__ && __F16C__
    if (is_avx_and_f16c()) {
  #pragma omp parallel for simd num_threads(_num_threads)
      for (size_t i = 0; i < (size_t)(len / 8) * 8; i += 8) {
        // convert to m128
        _mm_storeu_si128((__m128i*)(out+i), _mm256_cvtps_ph((__m256)_mm256_loadu_si256((__m256i*)(in + i)), 0));
      }
    }
    for (size_t i = (len / 8) * 8; i < (size_t)len; ++i) {
      float in_float = in[i];
      Float2HalfBits(&in_float, out+i);
    }
  #else
  #pragma omp parallel for simd num_threads(_num_threads)
      for (size_t i = 0; i < (size_t)len; ++i) {
        Float2HalfBits(&(in + i), out + i);
      }
  #endif
  }

tensor_t Compressor::FP16TensortoFP32(tensor_t grad) {
  assert(grad.dtype == BYTEPS_FLOAT16);
  size_t size = grad.size;
  if (_worker_fp16 = false) {
    _worker_fp16 = true;
  }
  FP16toFP32((void*)(_convert_buf.get()), (void*)(grad.data), size);
  return {_convert_buf.get(), size*2};
}

tensor_t Compressor::FP32TensortoFP16(tensor_t grad) {
  assert(grad.dtype == BYTEPS_FLOAT32);
  size_t size = grad.size;
  FP32toFP16((void*)(_buf.get()), (void*)(grad.data), size);
  std::memcpy(grad.data, _buf.get(), size/2);
  return {grad.data, size/2, BYTEPS_FLOAT16};
}


}  // namespace compressor
}  // namespace common
}  // namespace byteps