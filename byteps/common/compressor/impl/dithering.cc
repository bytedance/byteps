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

#include <cmath>
#include <cstring>

#include "../compressor_registry.h"
#include "dithering.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "dithering_compressor",
    [](const kwargs_t& kwargs, size_t size,
       DataType dtype) -> std::unique_ptr<Compressor> {
      std::tuple<> params;
      auto k = HyperParamFinder<unsigned>(kwargs, "compressor_k");

      auto seed = HyperParamFinder<unsigned>(kwargs, "seed", true,
                                             [](unsigned x) { return x != 0; });

      auto ptype_int = HyperParamFinder<int>(
          kwargs, "partition", true, [](int x) { return x == 0 || x == 1; });
      auto ptype = static_cast<DitheringCompressor::PartitionType>(ptype_int);

      auto ntype_int = HyperParamFinder<int>(
          kwargs, "normalize", true, [](int x) { return x == 0 || x == 1; });
      auto ntype = static_cast<DitheringCompressor::NomalizeType>(ntype_int);

      return std::unique_ptr<Compressor>(
          new DitheringCompressor(size, dtype, k, seed, ptype, ntype));
    });
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                           size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");

  // normalize
  double scale = 0.0;
  if (_ntype == NomalizeType::MAX) {
    for (size_t i = 0; i < len; i++) {
      scale = scale > std::abs(src[i]) ? scale : std::abs(src[i]);
    }
  } else if (_ntype == NomalizeType::L2) {
    for (size_t i = 0; i < len; ++i) {
      scale += src[i] * src[i];
    }
    scale = std::sqrt(scale);
  }

  BitWriter<index_t> bit_writer(dst);
  size_t last_non_zero_pos = -1;
  if (_ptype == PartitionType::LINEAR) {
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * _s;
      float floor = std::floor(normalized);
      unsigned quantized = floor + _rng.Bernoulli(normalized - floor);
      if (quantized) {
        size_t diff = i - last_non_zero_pos;
        last_non_zero_pos = i;
        EliasDeltaEncode(bit_writer, diff);
        bit_writer.Put(std::signbit(src[i]));
        EliasDeltaEncode(bit_writer, quantized);
      }
    }
  } else if (_ptype == PartitionType::NATURAL) {
    const unsigned level = 1 << (_s - 1);
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * level;
      float floor = RoundNextPow2(std::ceil(normalized)) << 1;
      unsigned quantized =
          floor * (1 + _rng.Bernoulli((normalized - floor) / floor));
      if (quantized) {
        size_t diff = i - last_non_zero_pos;
        last_non_zero_pos = i;
        EliasDeltaEncode(bit_writer, diff);
        bit_writer.Put(std::signbit(src[i]));
        EliasDeltaEncode(bit_writer, quantized);
      }
    }
  }
  bit_writer.Flush();

  // bits
  index_t* p_bits = reinterpret_cast<index_t*>(&dst[bit_writer.blocks()]);
  *p_bits = bit_writer.bits();

  // l2
  float* p_scale = reinterpret_cast<float*>(&dst[bit_writer.blocks() + 1]);
  *p_scale = scale;

  return {dst, bit_writer.blocks() * sizeof(index_t) + sizeof(index_t) +
                   sizeof(float)};
}

tensor_t DitheringCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                             size_t compressed_size) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");

  const size_t blocks =
      (compressed_size - sizeof(float) - sizeof(index_t)) / sizeof(index_t);
  auto* p_bits = reinterpret_cast<const index_t*>(src + blocks);
  const index_t bits = *p_bits;

  auto* p_scale = reinterpret_cast<const float*>(src + blocks + 1);
  const float scale = *p_scale;

  auto ptr = const_cast<index_t*>(src);
  if ((void*)dst == (void*)src) {
    ptr = reinterpret_cast<index_t*>(_buf.get());
    std::memcpy(ptr, src, compressed_size);
  }
  std::memset(dst, 0, _size);

  unsigned int s = _s;
  if (_ptype == PartitionType::NATURAL) {
    s = 1 << (_s - 1);
  }

  BitReader<index_t> bit_reader(ptr);
  size_t last_non_zero_pos = -1;
  while (bit_reader.bits() < bits) {
    size_t diff = EliasDeltaDecode(bit_reader);
    size_t i = last_non_zero_pos + diff;
    last_non_zero_pos = i;
    int signbit = bit_reader.Get();
    unsigned quantized = EliasDeltaDecode(bit_reader);
    float num = quantized * scale / s;
    dst[i] = (1 - (signbit << 1)) * num;
  }

  return {dst, _size};
}

tensor_t DitheringCompressor::Decompress(tensor_t compressed) {
#ifdef BYTEPS_BUILDING_SERVER
  auto dst = _buf.get();
#else
  auto dst = compressed.data;
#endif
  DECOMPRESS_IMPL_SWITCH(_dtype, DecompressImpl, dst, compressed.data,
                         compressed.size);
}

template <typename index_t, typename scalar_t>
void DitheringCompressor::FastUpdateErrorImpl(scalar_t* error,
                                              scalar_t* corrected,
                                              const index_t* compressed,
                                              size_t compressed_size) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");

  const size_t blocks =
      (compressed_size - sizeof(float) - sizeof(index_t)) / sizeof(index_t);
  auto* p_bits = reinterpret_cast<const index_t*>(compressed + blocks);
  const index_t bits = *p_bits;

  auto* p_scale = reinterpret_cast<const float*>(compressed + blocks + 1);
  const float scale = *p_scale;

  std::memcpy(error, corrected, _size);

  unsigned int s = _s;
  if (_ptype == PartitionType::NATURAL) {
    s = 1 << (_s - 1);
  }

  BitReader<index_t> bit_reader(compressed);
  size_t last_non_zero_pos = -1;
  while (bit_reader.bits() < bits) {
    size_t diff = EliasDeltaDecode(bit_reader);
    size_t i = last_non_zero_pos + diff;
    last_non_zero_pos = i;
    int signbit = bit_reader.Get();
    unsigned quantized = EliasDeltaDecode(bit_reader);
    float num = quantized * scale / s;
    error[i] -= (1 - (signbit << 1)) * num;
  }
}

void DitheringCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                          tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(_dtype, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps