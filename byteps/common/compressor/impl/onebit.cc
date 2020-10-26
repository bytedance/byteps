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

#include "onebit.h"

#include <cstring>

#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "onebit_compressor", [](const kwargs_t& kwargs, size_t size, DataType dtype,
                            std::unique_ptr<Compressor> cptr) {
      auto scaled =
          HyperParamFinder<bool>(kwargs, "compressor_onebit_scaling", true);
      BPS_LOG(INFO) << "onebit compressor is registered."
                    << "\tsize=" << size << "\tscaled=" << scaled;
      return std::unique_ptr<Compressor>(
          new OnebitCompressor(size, dtype, scaled));
    });
}

template <typename index_t, typename scalar_t>
size_t OnebitCompressor::CompressImpl(index_t* __restrict__ dst,
                                      const scalar_t* __restrict__ src,
                                      size_t len) {
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  const size_t chunk_len = (len + padding_len) / PACKING_SIZE;

  float scale = 1.0f;
  if (_use_scale) {
    double sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
    for (size_t i = 0; i < len; ++i) {
      sum += std::abs(src[i]);
    }
    scale = sum / len;
  }

#pragma omp parallel for simd
  for (size_t i = 0; i < chunk_len; ++i) {
    size_t idx = i * PACKING_SIZE;
    index_t x = src[idx] < 0;
    for (size_t j = 1; j < PACKING_SIZE; ++j) {
      x <<= 1;
      x |= src[idx + j] < 0;
    }
    dst[i] = x;
  }

  auto p_scale = reinterpret_cast<float*>(&dst[chunk_len]);
  *p_scale = scale;

  return chunk_len * sizeof(index_t) + sizeof(float);
}  // namespace compressor

void OnebitCompressor::Compress(tensor_t grad, tensor_t& output) {
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.data != output.data);
  if (output.data == nullptr) {
    output.data = _buf.get();
  }

  size_t compressed_size;
  switch (grad.dtype) {
    case BYTEPS_FLOAT16:
      compressed_size = CompressImpl(reinterpret_cast<int64_t*>(output.data),
                                     reinterpret_cast<const half_t*>(grad.data),
                                     grad.size / sizeof(half_t));
      break;
    case BYTEPS_FLOAT32:
      compressed_size = CompressImpl(reinterpret_cast<int64_t*>(output.data),
                                     reinterpret_cast<const float*>(grad.data),
                                     grad.size / sizeof(float));
      break;
    case BYTEPS_FLOAT64:
      compressed_size = CompressImpl(reinterpret_cast<int64_t*>(output.data),
                                     reinterpret_cast<const double*>(grad.data),
                                     grad.size / sizeof(double));
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
  }

  output.size = compressed_size;
}

template <typename scalar_t, typename index_t>
void OnebitCompressor::DecompressImpl(scalar_t* __restrict__ dst,
                                      const index_t* __restrict__ src,
                                      size_t compressed_size, size_t dst_size) {
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  const size_t chunk_len = (compressed_size - sizeof(float)) / sizeof(index_t);

  auto* pf = reinterpret_cast<const float*>(src + chunk_len);
  float scale = *pf;

#pragma omp parallel for simd
  for (int i = 0; i < chunk_len; ++i) {
    index_t x = src[i];
    size_t idx = i * PACKING_SIZE;
    for (int j = 0; j < PACKING_SIZE; ++j) {
      int sign = 1 - (x < 0) - (x < 0);
      dst[idx + j] = sign * scale;
      x <<= 1;
    }
  }
}

void OnebitCompressor::Decompress(tensor_t compressed, tensor_t& output) {
  BPS_CHECK(compressed.data);
  BPS_CHECK(compressed.data != output.data);

  if (output.data == nullptr) {
    output = {_buf.get(), _size, _dtype};
  } else {
    BPS_CHECK(output.size > 0);
  }

  switch (output.dtype) {
    case BYTEPS_FLOAT16:
      DecompressImpl(reinterpret_cast<half_t*>(output.data),
                     reinterpret_cast<const int64_t*>(compressed.data),
                     compressed.size, output.size);
      break;
    case BYTEPS_FLOAT32:
      DecompressImpl(reinterpret_cast<float*>(output.data),
                     reinterpret_cast<const int64_t*>(compressed.data),
                     compressed.size, output.size);
      break;
    case BYTEPS_FLOAT64:
      DecompressImpl(reinterpret_cast<double*>(output.data),
                     reinterpret_cast<const int64_t*>(compressed.data),
                     compressed.size, output.size);
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << output.dtype;
  }
}

template <typename index_t, typename scalar_t>
size_t OnebitCompressor::FusedCompressImpl(index_t* __restrict__ dst,
                                           const scalar_t* __restrict__ src,
                                           scalar_t* __restrict__ error,
                                           size_t len) {
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  const size_t chunk_len = (len + padding_len) / PACKING_SIZE;

  float scale = 1.0f;
  if (_use_scale) {
    double sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
    for (size_t i = 0; i < len; ++i) {
      sum += std::abs(src[i]);
    }
    scale = sum / len;
  }

#pragma omp parallel for simd
  for (size_t i = 0; i < chunk_len; ++i) {
    size_t idx = i * PACKING_SIZE;
    index_t x = src[idx] < 0;
    error[idx] = src[idx] - scale * (1 - x - x);
    for (size_t j = 1; j < PACKING_SIZE; ++j) {
      x <<= 1;
      index_t sign = src[idx + j] < 0;
      error[idx + j] = src[idx + j] - scale * (1 - sign - sign);
      x |= sign;
    }
    dst[i] = x;
  }

  auto p_scale = reinterpret_cast<float*>(&dst[chunk_len]);
  *p_scale = scale;

  return chunk_len * sizeof(index_t) + sizeof(float);
}  // namespace compressor

void OnebitCompressor::FusedCompress(tensor_t grad, tensor_t& output,
                                     tensor_t error) {
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.data != output.data);

  if (output.data == nullptr) {
    output.data = _buf.get();
  }

  size_t compressed_size;
  switch (grad.dtype) {
    case BYTEPS_FLOAT16:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<int64_t*>(output.data),
          reinterpret_cast<const half_t*>(grad.data),
          reinterpret_cast<half_t*>(error.data), grad.size / sizeof(half_t));
      break;
    case BYTEPS_FLOAT32:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<int64_t*>(output.data),
          reinterpret_cast<const float*>(grad.data),
          reinterpret_cast<float*>(error.data), grad.size / sizeof(float));
      break;
    case BYTEPS_FLOAT64:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<int64_t*>(output.data),
          reinterpret_cast<const double*>(grad.data),
          reinterpret_cast<double*>(error.data), grad.size / sizeof(double));
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
  }

  output.size = compressed_size;
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps