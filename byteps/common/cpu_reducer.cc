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
int CpuReducer::_sum(T* dst, const T* src, size_t len) {
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
int CpuReducer::_sum(T* dst, const T* src1, const T* src2, size_t len) {
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < len / (size_t)sizeof(T); ++i) {
    dst[i] = src1[i] + src2[i];
  }
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
      return _sum(reinterpret_cast<half_t*>(dst),
                  reinterpret_cast<const half_t*>(src), len, alpha);
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
      return _sum(reinterpret_cast<half_t*>(dst),
                  reinterpret_cast<const half_t*>(src1),
                  reinterpret_cast<const half_t*>(src2), len, alpha);
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

int CpuReducer::sparse_sum(void* dst, const void* src, size_t size,
                           DataType dtype, float alpha,
                           const std::vector<uint32_t>& idx_list) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return _sparse_sum(reinterpret_cast<float*>(dst),
                         reinterpret_cast<const float*>(src),
                         size / sizeof(float), alpha, idx_list);
    case BYTEPS_FLOAT64:
      return _sparse_sum(reinterpret_cast<double*>(dst),
                         reinterpret_cast<const double*>(src),
                         size / sizeof(double), alpha, idx_list);
    case BYTEPS_FLOAT16:
      return _sparse_sum(reinterpret_cast<half_t*>(dst),
                         reinterpret_cast<const half_t*>(src),
                         size / sizeof(half_t), alpha, idx_list);
    case BYTEPS_UINT8:
      return _sparse_sum(reinterpret_cast<uint8_t*>(dst),
                         reinterpret_cast<const uint8_t*>(src),
                         size / sizeof(uint8_t), alpha, idx_list);
    case BYTEPS_INT32:
      return _sparse_sum(reinterpret_cast<int32_t*>(dst),
                         reinterpret_cast<const int32_t*>(src),
                         size / sizeof(int32_t), alpha, idx_list);
    case BYTEPS_INT8:
      return _sparse_sum(reinterpret_cast<int8_t*>(dst),
                         reinterpret_cast<const int8_t*>(src),
                         size / sizeof(int8_t), alpha, idx_list);
    case BYTEPS_INT64:
      return _sparse_sum(reinterpret_cast<int64_t*>(dst),
                         reinterpret_cast<const int64_t*>(src),
                         size / sizeof(int64_t), alpha, idx_list);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

template <typename T>
int CpuReducer::_sparse_sum(T* dst, const T* src, size_t len, float alpha,
                            const std::vector<uint32_t>& idx_list) {
  size_t size = idx_list.size();

#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < size; ++i) {
    dst[i] += src[idx_list[i]] * alpha;
  }

  return 0;
}
}  // namespace common
}  // namespace byteps
