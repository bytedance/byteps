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

#if __AVX__ && __F16C__
#include <cpuid.h>
#include <immintrin.h>
#endif

#include <cstring>
#include <memory>
#include "common.h"
#include "logging.h"

#ifndef BYTEPS_BUILDING_SERVER
#include "communicator.h"
#else
typedef void BytePSComm;
#endif

#include <stdint.h>

namespace byteps {
namespace common {

class CpuReducer {
 public:
  CpuReducer(std::shared_ptr<BytePSComm> comm);
  ~CpuReducer() {
    if (_comm) _comm.reset();
    BPS_LOG(DEBUG) << "Clear CpuReducer";
  }

  int sum(void* dst, const void* src, size_t len, DataType dtype);
  int sum(void* dst, const void* src1, const void* src2, size_t len,
          DataType dtype);

  int sum(void* dst, const void* src, size_t len, DataType dtype, float alpha);
  int sum(void* dst, const void* src1, const void* src2, size_t len,
          DataType dtype, float alpha);

  int copy(void* dst, const void* src, size_t len);

#ifndef BYTEPS_BUILDING_SERVER
  bool isRoot();
  std::shared_ptr<BytePSComm> getComm() { return _comm; }
#endif

  DataType GetDataType(int dtype) { return static_cast<DataType>(dtype); }

 private:

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

  template <typename T>
  int _sum(T* dst, const T* src, size_t len);
  template <typename T>
  int _sum(T* dst, const T* src1, const T* src2, size_t len);

  int _sum_float16(void* dst, const void* src, size_t len);
  int _sum_float16(void* dst, const void* src1, const void* src2, size_t len);

  template <typename T>
  int _sum(T* dst, const T* src, size_t len, float alpha);

  template <typename T>
  int _sum(T* dst, const T* src1, const T* src2, size_t len, float alpha);

  int _sum_float16(void* dst, const void* src, size_t len, float alpha);
  int _sum_float16(void* dst, const void* src1, const void* src2, size_t len,
                   float alpha);

  float _convert_half_to_full_precision(uint16_t h);
  uint16_t _convert_full_to_half_precision(float f);

  std::shared_ptr<BytePSComm> _comm;
  int _num_threads;
  size_t _single_thread_threshold; 
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_CPU_REDUCER_H
