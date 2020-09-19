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
tensor_t OnebitCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                        size_t len) {
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  const size_t chunk_len = (len + padding_len) / PACKING_SIZE;

  float scale = 1.0f;
  if (_use_scale) {
    double sum = 0.0f;
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

  float* p_scale = reinterpret_cast<float*>(&dst[chunk_len]);
  *p_scale = scale;

  return {dst, chunk_len * sizeof(index_t) + sizeof(float)};
}  // namespace compressor

tensor_t OnebitCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename scalar_t, typename index_t>
tensor_t OnebitCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                          size_t compressed_size) {
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  const size_t chunk_len = (compressed_size - sizeof(float)) / sizeof(index_t);

  auto* pf = reinterpret_cast<const float*>(src + chunk_len);
  float scale = *pf;

  index_t* ptr = const_cast<index_t*>(src);
  if ((void*)dst == (void*)src) {
    ptr = reinterpret_cast<index_t*>(_buf.get());
    std::memcpy(ptr, src, compressed_size);
  }

#pragma omp parallel for simd
  for (int i = chunk_len - 1; i >= 0; --i) {
    index_t x = ptr[i];
    size_t idx = i * PACKING_SIZE;
    for (int j = PACKING_SIZE - 1; j >= 0; --j) {
      int sign = 1 - ((x & 0x01) << 1);
      dst[idx + j] = sign * scale;
      x >>= 1;
    }
  }

  return {dst, _size};
}

tensor_t OnebitCompressor::Decompress(tensor_t compressed) {
#ifdef BYTEPS_BUILDING_SERVER
  auto dst = _buf.get();
#else
  auto dst = compressed.data;
#endif
  DECOMPRESS_IMPL_SWITCH(_dtype, DecompressImpl, dst, compressed.data,
                         compressed.size);
}

template <typename scalar_t, typename index_t>
void OnebitCompressor::FastUpdateErrorImpl(scalar_t* error, scalar_t* corrected,
                                           const index_t* compressed,
                                           size_t compressed_size) {
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  const size_t chunk_len = (compressed_size - sizeof(float)) / sizeof(index_t);

  auto* pf = reinterpret_cast<const float*>(compressed + chunk_len);
  float scale = *pf;

#pragma omp parallel for simd
  for (int i = chunk_len - 1; i >= 0; --i) {
    index_t x = compressed[i];
    size_t idx = i * PACKING_SIZE;
    for (int j = PACKING_SIZE - 1; j >= 0; --j) {
      int sign = ((x & 0x01) << 1) - 1;
      error[idx + j] = corrected[idx + j] + sign * scale;
      x >>= 1;
    }
  }
}

void OnebitCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                       tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(_dtype, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps