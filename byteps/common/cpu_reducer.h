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
#include <vector>

#include "common.h"
#include "logging.h"

#ifndef BYTEPS_BUILDING_SERVER
#include "communicator.h"
#else
typedef void BytePSComm;
#endif

#include <cstdint>

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

  int sparse_sum(void* dst, const void* src, size_t size, DataType dtype,
                 float alpha, const std::vector<uint32_t>& idx_list);

  int copy(void* __restrict__ dst, const void* __restrict__ src, size_t len);

#ifndef BYTEPS_BUILDING_SERVER
  bool isRoot();
  std::shared_ptr<BytePSComm> getComm() { return _comm; }
#endif

  DataType GetDataType(int dtype) { return static_cast<DataType>(dtype); }

 private:
  template <typename T>
  int _sum(T* __restrict__ dst, const T* __restrict__ src, size_t len);

  template <typename T>
  int _sum(T* __restrict__ dst, const T* __restrict__ src1,
           const T* __restrict__ src2, size_t len);

  template <typename T>
  int _sum(T* __restrict__ dst, const T* __restrict__ src, size_t len,
           float alpha);

  template <typename T>
  int _sum(T* __restrict__ dst, const T* __restrict__ src1,
           const T* __restrict__ src2, size_t len, float alpha);

  template <typename T>
  int _sparse_sum(T* __restrict__ dst, const T* __restrict__ src, size_t len,
                  float alpha, const std::vector<uint32_t>& idx_list);

  float _convert_half_to_full_precision(uint16_t h);
  uint16_t _convert_full_to_half_precision(float f);

  std::shared_ptr<BytePSComm> _comm;
  int _num_threads;
  size_t _single_thread_threshold;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_CPU_REDUCER_H
