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

#ifndef BYTEPS_COMPRESSOR_COMMON_H
#define BYTEPS_COMPRESSOR_COMMON_H

#include <unordered_map>
#if __F16C__
#include "../half.h"
using half_t = mshadow::half::half_t;
#else
using half_t = void;
#endif

namespace byteps {
namespace common {
namespace compressor {
using byte_t = char;

struct BPSTensor {
  byte_t* data{nullptr};
  size_t size{0};
  int dtype{0};

  BPSTensor() = default;
  BPSTensor(void* data, size_t size = 0, int dtype = 0)
      : data(reinterpret_cast<byte_t*>(data)), size(size), dtype(dtype) {}
};
using tensor_t = BPSTensor;

using kwargs_t = std::unordered_map<std::string, std::string>;

#define COMPRESS_IMPL_SWITCH(dtype, func, dst, src, size)                     \
  switch (dtype) {                                                            \
    case BYTEPS_FLOAT16:                                                      \
      return func(reinterpret_cast<uint32_t*>(dst),                           \
                  reinterpret_cast<const half_t*>(src),                       \
                  size / sizeof(half_t));                                     \
    case BYTEPS_FLOAT32:                                                      \
      return func(reinterpret_cast<uint32_t*>(dst),                           \
                  reinterpret_cast<const float*>(src), size / sizeof(float)); \
    case BYTEPS_FLOAT64:                                                      \
      return func(reinterpret_cast<uint64_t*>(dst),                           \
                  reinterpret_cast<const double*>(src),                       \
                  size / sizeof(double));                                     \
    default:                                                                  \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                      \
  }

#define COMPRESS_IMPL_SWITCH2(dtype, func, dst, src, size)                    \
  switch (dtype) {                                                            \
    case BYTEPS_FLOAT16:                                                      \
      return func(reinterpret_cast<half_t*>(dst),                             \
                  reinterpret_cast<const half_t*>(src),                       \
                  size / sizeof(half_t));                                     \
    case BYTEPS_FLOAT32:                                                      \
      return func(reinterpret_cast<float*>(dst),                              \
                  reinterpret_cast<const float*>(src), size / sizeof(float)); \
    case BYTEPS_FLOAT64:                                                      \
      return func(reinterpret_cast<double*>(dst),                             \
                  reinterpret_cast<const double*>(src),                       \
                  size / sizeof(double));                                     \
    default:                                                                  \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                      \
  }

#define COMPRESS_IMPL_SCALAR_SWITCH(dtype, func, dst, src, size)              \
  switch (dtype) {                                                            \
    case BYTEPS_FLOAT16:                                                      \
      return func(reinterpret_cast<index_t*>(dst),                            \
                  reinterpret_cast<const half_t*>(src),                       \
                  size / sizeof(half_t));                                     \
    case BYTEPS_FLOAT32:                                                      \
      return func(reinterpret_cast<index_t*>(dst),                            \
                  reinterpret_cast<const float*>(src), size / sizeof(float)); \
    case BYTEPS_FLOAT64:                                                      \
      return func(reinterpret_cast<index_t*>(dst),                            \
                  reinterpret_cast<const double*>(src),                       \
                  size / sizeof(double));                                     \
    default:                                                                  \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                      \
  }

#define DECOMPRESS_IMPL_SWITCH(dtype, func, dst, src, compressed_size)      \
  switch (dtype) {                                                          \
    case BYTEPS_FLOAT16:                                                    \
      return func(reinterpret_cast<half_t*>(dst),                           \
                  reinterpret_cast<const uint32_t*>(src), compressed_size); \
    case BYTEPS_FLOAT32:                                                    \
      return func(reinterpret_cast<float*>(dst),                            \
                  reinterpret_cast<const uint32_t*>(src), compressed_size); \
    case BYTEPS_FLOAT64:                                                    \
      return func(reinterpret_cast<double*>(dst),                           \
                  reinterpret_cast<const uint64_t*>(src), compressed_size); \
    default:                                                                \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                    \
  }

#define DECOMPRESS_IMPL_SWITCH2(dtype, func, dst, src, compressed_size)  \
  switch (dtype) {                                                       \
    case BYTEPS_FLOAT16:                                                 \
      return func(reinterpret_cast<half_t*>(dst),                        \
                  reinterpret_cast<const float*>(src), compressed_size); \
    case BYTEPS_FLOAT32:                                                 \
      return func(reinterpret_cast<float*>(dst),                         \
                  reinterpret_cast<const float*>(src), compressed_size); \
    case BYTEPS_FLOAT64:                                                 \
      return func(reinterpret_cast<double*>(dst),                        \
                  reinterpret_cast<const float*>(src), compressed_size); \
    default:                                                             \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                 \
  }

#define DECOMPRESS_IMPL_SCALAR_SWITCH(dtype, func, dst, src, compressed_size) \
  switch (dtype) {                                                            \
    case BYTEPS_FLOAT16:                                                      \
      return func(reinterpret_cast<half_t*>(dst),                             \
                  reinterpret_cast<const index_t*>(src), compressed_size);    \
    case BYTEPS_FLOAT32:                                                      \
      return func(reinterpret_cast<float*>(dst),                              \
                  reinterpret_cast<const index_t*>(src), compressed_size);    \
    case BYTEPS_FLOAT64:                                                      \
      return func(reinterpret_cast<double*>(dst),                             \
                  reinterpret_cast<const index_t*>(src), compressed_size);    \
    default:                                                                  \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                      \
  }

#define FAST_UPDATE_ERROR_IMPL_SWITCH(dtype, func, dst, src1, src2,          \
                                      compressed_size)                       \
  switch (dtype) {                                                           \
    case BYTEPS_FLOAT16:                                                     \
      return func(reinterpret_cast<half_t*>(dst),                            \
                  reinterpret_cast<half_t*>(src1),                           \
                  reinterpret_cast<const uint32_t*>(src2), compressed_size); \
    case BYTEPS_FLOAT32:                                                     \
      return func(reinterpret_cast<float*>(dst),                             \
                  reinterpret_cast<float*>(src1),                            \
                  reinterpret_cast<const uint32_t*>(src2), compressed_size); \
    case BYTEPS_FLOAT64:                                                     \
      return func(reinterpret_cast<double*>(dst),                            \
                  reinterpret_cast<double*>(src1),                           \
                  reinterpret_cast<const uint64_t*>(src2), compressed_size); \
    default:                                                                 \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                     \
  }

#define FAST_UPDATE_ERROR_IMPL_SWITCH2(dtype, func, dst, src1, src2,       \
                                       compressed_size)                    \
  switch (dtype) {                                                         \
    case BYTEPS_FLOAT16:                                                   \
      return func(reinterpret_cast<half_t*>(dst),                          \
                  reinterpret_cast<half_t*>(src1),                         \
                  reinterpret_cast<const half_t*>(src2), compressed_size); \
    case BYTEPS_FLOAT32:                                                   \
      return func(reinterpret_cast<float*>(dst),                           \
                  reinterpret_cast<float*>(src1),                          \
                  reinterpret_cast<const float*>(src2), compressed_size);  \
    case BYTEPS_FLOAT64:                                                   \
      return func(reinterpret_cast<double*>(dst),                          \
                  reinterpret_cast<double*>(src1),                         \
                  reinterpret_cast<const double*>(src2), compressed_size); \
    default:                                                               \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                   \
  }

#define FAST_UPDATE_ERROR_IMPL_SCALAR_SWITCH(dtype, func, dst, src1, src2,  \
                                             compressed_size)               \
  switch (dtype) {                                                          \
    case BYTEPS_FLOAT16:                                                    \
      return func(reinterpret_cast<half_t*>(dst),                           \
                  reinterpret_cast<half_t*>(src1),                          \
                  reinterpret_cast<const index_t*>(src2), compressed_size); \
    case BYTEPS_FLOAT32:                                                    \
      return func(reinterpret_cast<float*>(dst),                            \
                  reinterpret_cast<float*>(src1),                           \
                  reinterpret_cast<const index_t*>(src2), compressed_size); \
    case BYTEPS_FLOAT64:                                                    \
      return func(reinterpret_cast<double*>(dst),                           \
                  reinterpret_cast<double*>(src1),                          \
                  reinterpret_cast<const index_t*>(src2), compressed_size); \
    default:                                                                \
      BPS_CHECK(0) << "Unsupported data type:" << dtype;                    \
  }

}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_COMMON_H