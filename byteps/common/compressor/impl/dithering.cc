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

#include "dithering.h"

#include <cmath>
#include <cstring>

#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "dithering_compressor",
    [](const kwargs_t& kwargs, size_t size, DataType dtype,
       std::unique_ptr<Compressor> cptr) -> std::unique_ptr<Compressor> {
      std::tuple<> params;
      auto k = HyperParamFinder<unsigned>(kwargs, "compressor_k");

      auto seed = HyperParamFinder<unsigned>(kwargs, "seed", true,
                                             [](unsigned x) { return x != 0; });

      auto ptype_int =
          HyperParamFinder<int>(kwargs, "dithering_partition", true,
                                [](int x) { return x == 0 || x == 1; });
      auto ptype = static_cast<DitheringCompressor::PartitionType>(ptype_int);

      auto ntype_int =
          HyperParamFinder<int>(kwargs, "dithering_normalize", true,
                                [](int x) { return x == 0 || x == 1; });
      auto ntype = static_cast<DitheringCompressor::NormalizeType>(ntype_int);

      BPS_LOG(INFO) << "dithering compressor is registered."
                    << "\tsize=" << size << "\tk=" << k << "\tseed=" << seed
                    << "\ttype" << ptype_int << "\tntype" << ntype_int;
      return std::unique_ptr<Compressor>(
          new DitheringCompressor(size, dtype, k, seed, ptype, ntype));
    });
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::CompressImplL2(index_t* __restrict__ dst,
                                             const scalar_t* __restrict__ src,
                                             size_t len) {
  // normalize
  double scale = 0.0;
  for (size_t i = 0; i < len; ++i) {
    scale += src[i] * src[i];
  }
  scale = std::sqrt(scale);
  const uint64_t MAX = std::numeric_limits<uint64_t>::max();

  BitWriter<index_t> bit_writer(dst);
  size_t last_non_zero_pos = -1;  // it's not a bug here...
  if (_ptype == PartitionType::LINEAR) {
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * _s;
      float floor = std::floor(normalized);
      double p = normalized - floor;
      int bernoulli = _rand_list[i] < p * MAX;
      unsigned quantized = floor + bernoulli;
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
      double normalized = (abs_x / scale) * level;
      unsigned floor = RoundNextPow2(std::ceil(normalized)) >> 1;
      unsigned length = (floor != 0) ? floor : 1;
      double p = (normalized - floor) / length;
      int bernoulli = _rand_list[i] < p * MAX;
      unsigned quantized = floor + length * bernoulli;
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
  auto p_bits = reinterpret_cast<index_t*>(&dst[bit_writer.blocks()]);
  *p_bits = bit_writer.bits();

  // l2
  auto p_scale = reinterpret_cast<float*>(&dst[bit_writer.blocks() + 1]);
  *p_scale = scale;

  return {dst, bit_writer.blocks() * sizeof(index_t) + sizeof(index_t) +
                   sizeof(float)};
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::CompressImplMax(index_t* __restrict__ dst,
                                              const scalar_t* __restrict__ src,
                                              size_t len) {
  double scale = 0.0;
  for (size_t i = 0; i < len; i++) {
    scale = scale > std::abs(src[i]) ? scale : std::abs(src[i]);
  }
  const uint64_t MAX = std::numeric_limits<uint64_t>::max();

  if (_ptype == PartitionType::LINEAR) {
    // #pragma omp parallel for simd
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * _s;
      float floor = std::floor(normalized);
      double p = normalized - floor;
      int bernoulli = _rng.Bernoulli(p);
      index_t quantized = floor + bernoulli;
      dst[i] = sgn(src[i]) * quantized;
    }
  } else if (_ptype == PartitionType::NATURAL) {
    const unsigned level = 1 << (_s - 1);
#pragma omp parallel for simd
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      double normalized = (abs_x / scale) * level;
      unsigned floor = RoundNextPow2(std::ceil(normalized)) >> 1;
      unsigned length = (floor != 0) ? floor : 1;
      double p = (normalized - floor) / length;
      int bernoulli = _rand_list[i] < p * MAX;
      index_t quantized = floor + length * bernoulli;
      dst[i] = sgn(src[i]) * quantized;
    }
  }

  auto ptr = reinterpret_cast<float*>(&dst[len]);
  *ptr = scale;

  return {dst, len * sizeof(index_t) + sizeof(float)};
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::CompressImpl(index_t* __restrict__ dst,
                                           const scalar_t* __restrict__ src,
                                           size_t len) {
  // for (size_t i = 0; i < len; ++i) {
  //   _rand_list.push_back(_rng.xorshift128p());
  // }
  if (std::is_same<index_t, int8_t>::value ||
      std::is_same<index_t, int16_t>::value) {
    return CompressImplMax<index_t, scalar_t>(dst, src, len);
  } else {
    return CompressImplL2<index_t, scalar_t>(dst, src, len);
  }
  // _rand_list.clear();
}

tensor_t DitheringCompressor::Compress(tensor_t grad) {
  switch (this->_ntype) {
    case NormalizeType::L2: {
      COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                           grad.size);
    } break;
    case NormalizeType::MAX: {
      if (this->_s < (1 << 7)) {
        using index_t = int8_t;
        COMPRESS_IMPL_SCALAR_SWITCH(grad.dtype, CompressImpl, _buf.get(),
                                    grad.data, grad.size);
      } else if (this->_s < (1 << 15)) {
        using index_t = int16_t;
        COMPRESS_IMPL_SCALAR_SWITCH(grad.dtype, CompressImpl, _buf.get(),
                                    grad.data, grad.size);
      } else {
        BPS_CHECK(0) << "k exceeds the maximum limit.";
      }
    } break;
    default:
      BPS_CHECK(0) << "Unsupport ntype";
  }
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::DecompressImpl(scalar_t* __restrict__ dst,
                                             const index_t* __restrict__ src,
                                             size_t compressed_size) {
  if (std::is_same<index_t, int8_t>::value ||
      std::is_same<index_t, int16_t>::value) {
    return DecompressImplMax<index_t, scalar_t>(dst, src, compressed_size);
  } else {
    return DecompressImplL2<index_t, scalar_t>(dst, src, compressed_size);
  }
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::DecompressImplL2(scalar_t* __restrict__ dst,
                                               const index_t* __restrict__ src,
                                               size_t compressed_size) {
  const size_t blocks =
      (compressed_size - sizeof(float) - sizeof(index_t)) / sizeof(index_t);
  auto* p_bits = reinterpret_cast<const index_t*>(src + blocks);
  const index_t bits = *p_bits;

  auto* p_scale = reinterpret_cast<const float*>(src + blocks + 1);
  const float scale = *p_scale;

  std::memset(dst, 0, _size);

  unsigned int s = _s;
  if (_ptype == PartitionType::NATURAL) {
    s = 1 << (_s - 1);
  }

  BitReader<index_t> bit_reader(src);
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

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::DecompressImplMax(scalar_t* __restrict__ dst,
                                                const index_t* __restrict__ src,
                                                size_t compressed_size) {
  size_t len = (compressed_size - sizeof(float)) / sizeof(index_t);
  auto* p_scale = reinterpret_cast<const float*>(src + len);
  const float scale = *p_scale;

  unsigned int s = _s;
  if (_ptype == PartitionType::NATURAL) {
    s = 1 << (_s - 1);
  }

#pragma omp parallel for simd
  for (int i = 0; i < len; ++i) {
    dst[i] = src[i] * scale / s;
  }

  return {dst, _size};
}

tensor_t DitheringCompressor::Decompress(tensor_t compressed) {
#ifdef BYTEPS_BUILDING_SERVER
  auto src = compressed.data;
  auto dst = _buf.get();
#else
  auto src = _buf.get();
  auto dst = compressed.data;
#endif
  switch (this->_ntype) {
    case NormalizeType::L2: {
      DECOMPRESS_IMPL_SWITCH(_dtype, DecompressImpl, dst, src, compressed.size);
    } break;
    case NormalizeType::MAX: {
      if (this->_s <= (1 << 7)) {
        using index_t = int8_t;
        DECOMPRESS_IMPL_SCALAR_SWITCH(_dtype, DecompressImpl, dst, src,
                                      compressed.size);
      } else if (this->_s <= (1 << 15)) {
        using index_t = int16_t;
        DECOMPRESS_IMPL_SCALAR_SWITCH(_dtype, DecompressImpl, dst, src,
                                      compressed.size);
      } else {
        BPS_CHECK(0) << "k exceeds the maximum limit.";
      }
    } break;
    default:
      BPS_CHECK(0) << "Unsupport ntype";
  }
}

template <typename index_t, typename scalar_t>
void DitheringCompressor::FastUpdateErrorImplL2(
    scalar_t* __restrict__ error, scalar_t* __restrict__ corrected,
    const index_t* __restrict__ compressed, size_t compressed_size) {
  const size_t blocks =
      (compressed_size - sizeof(float) - sizeof(index_t)) / sizeof(index_t);
  auto* p_bits = reinterpret_cast<const index_t*>(compressed + blocks);
  const index_t bits = *p_bits;

  auto* p_scale = reinterpret_cast<const float*>(compressed + blocks + 1);
  const float scale = *p_scale;

  memcpy_multithread(error, corrected, _size);

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

template <typename index_t, typename scalar_t>
void DitheringCompressor::FastUpdateErrorImplMax(
    scalar_t* __restrict__ error, scalar_t* __restrict__ corrected,
    const index_t* __restrict__ compressed, size_t compressed_size) {
  size_t len = (compressed_size - sizeof(float)) / sizeof(index_t);
  auto* p_scale = reinterpret_cast<const float*>(compressed + len);
  const float scale = *p_scale;

  memcpy_multithread(error, corrected, _size);

  unsigned int s = _s;
  if (_ptype == PartitionType::NATURAL) {
    s = 1 << (_s - 1);
  }

#pragma omp parallel for simd
  for (int i = 0; i < len; ++i) {
    error[i] -= compressed[i] * scale / s;
  }
}

template <typename index_t, typename scalar_t>
void DitheringCompressor::FastUpdateErrorImpl(
    scalar_t* __restrict__ error, scalar_t* __restrict__ corrected,
    const index_t* __restrict__ compressed, size_t compressed_size) {
  if (std::is_same<index_t, int8_t>::value ||
      std::is_same<index_t, int16_t>::value) {
    FastUpdateErrorImplMax<index_t, scalar_t>(error, corrected, compressed,
                                              compressed_size);
  } else {
    FastUpdateErrorImplL2<index_t, scalar_t>(error, corrected, compressed,
                                             compressed_size);
  }
}

void DitheringCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                          tensor_t compressed) {
  switch (this->_ntype) {
    case NormalizeType::L2: {
      FAST_UPDATE_ERROR_IMPL_SWITCH(_dtype, FastUpdateErrorImpl, error.data,
                                    corrected.data, compressed.data,
                                    compressed.size);
    } break;
    case NormalizeType::MAX: {
      if (this->_s <= (1 << 7)) {
        using index_t = int8_t;
        FAST_UPDATE_ERROR_IMPL_SCALAR_SWITCH(_dtype, FastUpdateErrorImpl,
                                             error.data, corrected.data,
                                             compressed.data, compressed.size);
      } else if (this->_s <= (1 << 15)) {
        using index_t = int16_t;
        FAST_UPDATE_ERROR_IMPL_SCALAR_SWITCH(_dtype, FastUpdateErrorImpl,
                                             error.data, corrected.data,
                                             compressed.data, compressed.size);
      } else {
        BPS_CHECK(0) << "k exceeds the maximum limit.";
      }
    } break;
    default:
      BPS_CHECK(0) << "Unsupport ntype";
  }
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps