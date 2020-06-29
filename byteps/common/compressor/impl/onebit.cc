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
#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg("onebit_compressor", [](const kwargs_t& kwargs,
                                                         size_t size,
                                                         DataType dtype) {
  BPS_LOG(DEBUG) << "Register Onebit Compressor";
  bool scaled = false;
  auto iter = kwargs.find("compressor_onebit_scaling");
  if (iter != kwargs.end()) {
    if (iter->second == "true" || iter->second == "True") scaled = true;
  }
  return std::unique_ptr<Compressor>(new OnebitCompressor(size, dtype, scaled));
});
}

template <typename index_t, typename scalar_t>
tensor_t OnebitCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                        size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  constexpr size_t PACKING_SIZE = sizeof(scalar_t) * 8;

  float scale = 1.0f;
  if (_use_scale) {
    double sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
      dst[i] = src[i] < 0;
      sum += abs(src[i]);
    }
    scale = sum / len;
  } else {
    for (size_t i = 0; i < len; ++i) {
      dst[i] = src[i] < 0;
    }
  }

  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  const size_t chunk_len = (len + padding_len) / PACKING_SIZE;

  for (int i = 1; i < PACKING_SIZE; ++i) {
    for (int j = 0; j < chunk_len; ++j) {
      dst[j] <<= 1;
      dst[j] |= dst[i * chunk_len + j] & 0x01;
    }
  }

  float* p_scale = reinterpret_cast<float*>(&dst[chunk_len]);
  *p_scale = scale;

  return {dst, chunk_len * sizeof(index_t) + sizeof(float)};
}

tensor_t OnebitCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename scalar_t, typename index_t>
tensor_t OnebitCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                          size_t compressed_size) {
  static_assert(sizeof(scalar_t) == sizeof(index_t),
                "scalar_t should be the same size as index_t");
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  const size_t chunk_len = (compressed_size - sizeof(float)) / sizeof(index_t);

  auto* pf = reinterpret_cast<const float*>(src + chunk_len);
  float scale = *pf;

  auto ptr = reinterpret_cast<index_t*>(dst);
  if ((void*)dst != (void*)src) {
    std::copy(src, src + chunk_len, ptr);
  }

  for (int i = PACKING_SIZE - 1; i >= 1; --i) {
    for (int j = 0; j < chunk_len; ++j) {
      int sign = -(((ptr[j] & 0x01) << 1) - 1);
      dst[i * chunk_len + j] = sign * scale;
      ptr[j] >>= 1;
    }
  }

  // for i = 0 chunk
  for (int j = 0; j < chunk_len; ++j) {
    int sign = -(((ptr[j] & 0x01) << 1) - 1);
    dst[j] = sign * scale;
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

  std::memcpy(error, compressed, chunk_len * sizeof(index_t));

  auto ptr = reinterpret_cast<index_t*>(error);
  for (int i = PACKING_SIZE - 1; i >= 1; --i) {
    for (int j = 0; j < chunk_len; ++j) {
      int sign = ((ptr[j] & 0x01) << 1) - 1;
      error[i * chunk_len + j] = corrected[i * chunk_len + j] + sign * scale;
      ptr[j] >>= 1;
    }
  }

  // for i = 0 chunk
  for (int j = 0; j < chunk_len; ++j) {
    int sign = ((ptr[j] & 0x01) << 1) - 1;
    error[j] = corrected[j] + sign * scale;
  }
}

void OnebitCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                       tensor_t compressed) {
  SWITCH_TO_FAST_UPDATE_ERROR_IMPL_SWITCH(_dtype, FastUpdateErrorImpl,
                                          error.data, corrected.data,
                                          compressed.data, compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps