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
#if __F16C__
#include "half.h"
using half_t = mshadow::half::half_t;
#else
using half_t = void;
#endif

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
      return _sum(reinterpret_cast<half_t*>(dst),
                  reinterpret_cast<const half_t*>(src), len);
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
int CpuReducer::_sum(T* __restrict__ dst, const T* __restrict__ src,
                     size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] = dst[i] + src[i];
  }
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
      return _sum(reinterpret_cast<half_t*>(dst),
                  reinterpret_cast<const half_t*>(src1),
                  reinterpret_cast<const half_t*>(src2), len);
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
int CpuReducer::_sum(T* __restrict__ dst, const T* __restrict__ src1,
                     const T* __restrict__ src2, size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] = src1[i] + src2[i];
  }
  return 0;
}

int CpuReducer::copy(void* __restrict__ dst, const void* __restrict__ src,
                     size_t len) {
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

int CpuReducer::sum_mixed_precision(void* dst, const void* src, size_t len,
                                    DataType dtype) {
  switch (dtype) {
    case BYTEPS_FLOAT16:
      return _sum_mixed_precision(reinterpret_cast<float*>(dst),
                                  reinterpret_cast<const half_t*>(src), len);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
}

template <typename T>
int CpuReducer::_sum_mixed_precision(float* __restrict__ dst,
                                     const T* __restrict__ src, size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] += src[i];
  }
  return 0;
}

int CpuReducer::copy_mixed_precision(void* dst, const void* src, size_t len,
                                     DataType dtype, bool up) {
  if (up) {
    // cast into higher data type
    switch (dtype) {
      case BYTEPS_FLOAT16:
        return _copy_mixed_precision_up(reinterpret_cast<float*>(dst),
                                        reinterpret_cast<const half_t*>(src),
                                        len);
      default:
        BPS_CHECK(0) << "Unsupported data type: " << dtype;
    }
  } else {
    // cast into lower data type
    switch (dtype) {
      case BYTEPS_FLOAT16:
        return _copy_mixed_precision_down(reinterpret_cast<half_t*>(dst),
                                          reinterpret_cast<const float*>(src),
                                          len);
      default:
        BPS_CHECK(0) << "Unsupported data type: " << dtype;
    }
  }
}

template <typename T>
int CpuReducer::_copy_mixed_precision_up(float* __restrict__ dst,
                                         const T* __restrict__ src,
                                         size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / sizeof(T); ++i) {
    dst[i] = src[i];
  }
  return 0;
}

template <typename T>
int CpuReducer::_copy_mixed_precision_down(T* __restrict__ dst,
                                           const float* __restrict__ src,
                                           size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / sizeof(T); ++i) {
    dst[i] = src[i];
  }
  return 0;
}
}  // namespace common
}  // namespace byteps
