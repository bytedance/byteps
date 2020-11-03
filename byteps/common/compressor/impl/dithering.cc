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
size_t DitheringCompressor::CompressImplL2(index_t* __restrict__ dst,
                                           const scalar_t* __restrict__ src,
                                           size_t len) {
  // normalize
  double scale = 0.0;
#pragma omp parallel for simd reduction(+ : scale)
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
      float p = normalized - floor;
      int bernoulli = static_cast<float>(_rand_list[i]) < p * MAX;
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
      float normalized = (abs_x / scale) * level;
      unsigned floor = RoundNextPow2(std::ceil(normalized)) >> 1;
      unsigned length = (floor != 0) ? floor : 1;
      float p = (normalized - floor) / length;
      int bernoulli = static_cast<float>(_rand_list[i]) < p * MAX;
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

  return bit_writer.blocks() * sizeof(index_t) + sizeof(index_t) +
         sizeof(float);
}

template <typename index_t, typename scalar_t>
size_t DitheringCompressor::CompressImplMax(index_t* __restrict__ dst,
                                            const scalar_t* __restrict__ src,
                                            size_t len) {
  double scale = 0.0;
#pragma omp parallel for simd reduction(max : scale)
  for (size_t i = 0; i < len; i++) {
    scale = scale > std::fabs(src[i]) ? scale : std::fabs(src[i]);
  }
  const uint64_t MAX = std::numeric_limits<uint64_t>::max();

  if (_ptype == PartitionType::LINEAR) {
#pragma omp parallel for simd
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::fabs(src[i]);
      float normalized = (abs_x * _s) / scale;
      float floor = std::floor(normalized);
      float p = normalized - floor;
      // int bernoulli = static_cast<float>(_rand_list[i]) < p * MAX;
      // index_t quantized = floor + bernoulli;
      index_t quantized = floor;
      dst[i] = sgn(src[i]) * quantized;
    }
  } else if (_ptype == PartitionType::NATURAL) {
    const unsigned level = 1 << (_s - 1);
#pragma omp parallel for simd
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * level;
      unsigned floor = RoundNextPow2(std::ceil(normalized)) >> 1;
      unsigned length = (floor != 0) ? floor : 1;
      float p = (normalized - floor) / length;
      int bernoulli = static_cast<float>(_rand_list[i]) < p * MAX;
      index_t quantized = floor + length * bernoulli;
      dst[i] = sgn(src[i]) * quantized;
    }
  }

  auto ptr = reinterpret_cast<float*>(&dst[len]);
  *ptr = scale;

  return len * sizeof(index_t) + sizeof(float);
}

template <typename index_t, typename scalar_t>
size_t DitheringCompressor::CompressImpl(index_t* __restrict__ dst,
                                         const scalar_t* __restrict__ src,
                                         size_t len) {
  _rand_list.clear();
  for (size_t i = 0; i < len; ++i) {
    _rand_list.push_back(_rng.xorshift128p());
  }
  if (std::is_same<index_t, int8_t>::value ||
      std::is_same<index_t, int16_t>::value) {
    return CompressImplMax<index_t, scalar_t>(dst, src, len);
  } else {
    return CompressImplL2<index_t, scalar_t>(dst, src, len);
  }
}

void DitheringCompressor::Compress(tensor_t grad, tensor_t& output) {
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.data != output.data);
  if (output.data == nullptr) {
    output.data = _buf.get();
  }

  size_t compressed_size;
  if (this->_ntype == NormalizeType::L2) {
    switch (grad.dtype) {
      case BYTEPS_FLOAT16:
        compressed_size =
            CompressImpl(reinterpret_cast<uint32_t*>(output.data),
                         reinterpret_cast<const half_t*>(grad.data),
                         grad.size / sizeof(half_t));
        break;
      case BYTEPS_FLOAT32:
        compressed_size =
            CompressImpl(reinterpret_cast<uint32_t*>(output.data),
                         reinterpret_cast<const float*>(grad.data),
                         grad.size / sizeof(float));
        break;
      case BYTEPS_FLOAT64:
        compressed_size =
            CompressImpl(reinterpret_cast<uint32_t*>(output.data),
                         reinterpret_cast<const double*>(grad.data),
                         grad.size / sizeof(double));
        break;
      default:
        BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
    }
  } else if (this->_ntype == NormalizeType::MAX) {
    if (this->_s < (1 << 7)) {
      switch (grad.dtype) {
        case BYTEPS_FLOAT16:
          compressed_size =
              CompressImpl(reinterpret_cast<int8_t*>(output.data),
                           reinterpret_cast<const half_t*>(grad.data),
                           grad.size / sizeof(half_t));
          break;
        case BYTEPS_FLOAT32:
          compressed_size =
              CompressImpl(reinterpret_cast<int8_t*>(output.data),
                           reinterpret_cast<const float*>(grad.data),
                           grad.size / sizeof(float));
          break;
        case BYTEPS_FLOAT64:
          compressed_size =
              CompressImpl(reinterpret_cast<int8_t*>(output.data),
                           reinterpret_cast<const double*>(grad.data),
                           grad.size / sizeof(double));
          break;
        default:
          BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
      }
    } else if (this->_s < (1 << 15)) {
      switch (grad.dtype) {
        case BYTEPS_FLOAT16:
          compressed_size =
              CompressImpl(reinterpret_cast<int16_t*>(output.data),
                           reinterpret_cast<const half_t*>(grad.data),
                           grad.size / sizeof(half_t));
          break;
        case BYTEPS_FLOAT32:
          compressed_size =
              CompressImpl(reinterpret_cast<int16_t*>(output.data),
                           reinterpret_cast<const float*>(grad.data),
                           grad.size / sizeof(float));
          break;
        case BYTEPS_FLOAT64:
          compressed_size =
              CompressImpl(reinterpret_cast<int16_t*>(output.data),
                           reinterpret_cast<const double*>(grad.data),
                           grad.size / sizeof(double));
          break;
        default:
          BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
      }
    } else {
      BPS_CHECK(0) << "k exceeds the maximum limit.";
    }
  } else {
    BPS_CHECK(0) << "unsupport ntype";
  }

  output.size = compressed_size;
}

template <typename scalar_t, typename index_t>
void DitheringCompressor::DecompressImpl(scalar_t* __restrict__ dst,
                                         const index_t* __restrict__ src,
                                         size_t compressed_size,
                                         size_t dst_size) {
  if (std::is_same<index_t, int8_t>::value ||
      std::is_same<index_t, int16_t>::value) {
    return DecompressImplMax<scalar_t, index_t>(dst, src, compressed_size,
                                                dst_size);
  } else {
    return DecompressImplL2<scalar_t, index_t>(dst, src, compressed_size,
                                               dst_size);
  }
}

template <typename scalar_t, typename index_t>
void DitheringCompressor::DecompressImplL2(scalar_t* __restrict__ dst,
                                           const index_t* __restrict__ src,
                                           size_t compressed_size,
                                           size_t dst_size) {
  const size_t blocks =
      (compressed_size - sizeof(float) - sizeof(index_t)) / sizeof(index_t);
  auto* p_bits = reinterpret_cast<const index_t*>(src + blocks);
  const index_t bits = *p_bits;

  auto* p_scale = reinterpret_cast<const float*>(src + blocks + 1);
  const float scale = *p_scale;

  std::memset(dst, 0, dst_size);

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
}

template <typename scalar_t, typename index_t>
void DitheringCompressor::DecompressImplMax(scalar_t* __restrict__ dst,
                                            const index_t* __restrict__ src,
                                            size_t compressed_size,
                                            size_t dst_size) {
  size_t len = (compressed_size - sizeof(float)) / sizeof(index_t);
  auto* p_scale = reinterpret_cast<const float*>(src + len);
  const float scale = *p_scale;

  unsigned int s = _s;
  if (_ptype == PartitionType::NATURAL) {
    s = 1 << (_s - 1);
  }

#pragma omp parallel for simd
  for (int i = 0; i < len; ++i) {
    float num = src[i] * scale / s;
    dst[i] = num;
  }
}

void DitheringCompressor::Decompress(tensor_t compressed, tensor_t& output) {
  BPS_CHECK(compressed.data);
  BPS_CHECK(compressed.data != output.data);

  if (output.data == nullptr) {
    output = {_buf.get(), _size, _dtype};
  } else {
    BPS_CHECK(output.size > 0);
  }

  if (this->_ntype == NormalizeType::L2) {
    switch (output.dtype) {
      case BYTEPS_FLOAT16:
        DecompressImpl(reinterpret_cast<half_t*>(output.data),
                       reinterpret_cast<const uint32_t*>(compressed.data),
                       compressed.size, output.size);
        break;
      case BYTEPS_FLOAT32:
        DecompressImpl(reinterpret_cast<float*>(output.data),
                       reinterpret_cast<const uint32_t*>(compressed.data),
                       compressed.size, output.size);
        break;
      case BYTEPS_FLOAT64:
        DecompressImpl(reinterpret_cast<double*>(output.data),
                       reinterpret_cast<const uint32_t*>(compressed.data),
                       compressed.size, output.size);
        break;
      default:
        BPS_CHECK(0) << "Unsupported data type:" << output.dtype;
    }
  } else if (this->_ntype == NormalizeType::MAX) {
    if (this->_s < (1 << 7)) {
      switch (output.dtype) {
        case BYTEPS_FLOAT16:
          DecompressImpl(reinterpret_cast<half_t*>(output.data),
                         reinterpret_cast<const int8_t*>(compressed.data),
                         compressed.size, output.size);
          break;
        case BYTEPS_FLOAT32:
          DecompressImpl(reinterpret_cast<float*>(output.data),
                         reinterpret_cast<const int8_t*>(compressed.data),
                         compressed.size, output.size);
          break;
        case BYTEPS_FLOAT64:
          DecompressImpl(reinterpret_cast<double*>(output.data),
                         reinterpret_cast<const int8_t*>(compressed.data),
                         compressed.size, output.size);
          break;
        default:
          BPS_CHECK(0) << "Unsupported data type:" << output.dtype;
      }
    } else if (this->_s < (1 << 15)) {
      switch (output.dtype) {
        case BYTEPS_FLOAT16:
          DecompressImpl(reinterpret_cast<half_t*>(output.data),
                         reinterpret_cast<const int16_t*>(compressed.data),
                         compressed.size, output.size);
          break;
        case BYTEPS_FLOAT32:
          DecompressImpl(reinterpret_cast<float*>(output.data),
                         reinterpret_cast<const int16_t*>(compressed.data),
                         compressed.size, output.size);
          break;
        case BYTEPS_FLOAT64:
          DecompressImpl(reinterpret_cast<double*>(output.data),
                         reinterpret_cast<const int16_t*>(compressed.data),
                         compressed.size, output.size);
          break;
        default:
          BPS_CHECK(0) << "Unsupported data type:" << output.dtype;
      }
    } else {
      BPS_CHECK(0) << "k exceeds the maximum limit.";
    }
  } else {
    BPS_CHECK(0) << "unsupport ntype";
  }
}

template <typename index_t, typename scalar_t>
size_t DitheringCompressor::FusedCompressImplL2(
    index_t* __restrict__ dst, const scalar_t* __restrict__ src,
    scalar_t* __restrict__ error, size_t len) {
  // normalize
  double scale = 0.0;
#pragma omp parallel for simd reduction(+ : scale)
  for (size_t i = 0; i < len; ++i) {
    scale += src[i] * src[i];
  }
  scale = std::sqrt(scale);
  const uint64_t MAX = std::numeric_limits<uint64_t>::max();

  memcpy_multithread(error, src, len * sizeof(scalar_t));

  BitWriter<index_t> bit_writer(dst);
  size_t last_non_zero_pos = -1;  // it's not a bug here...
  if (_ptype == PartitionType::LINEAR) {
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * _s;
      float floor = std::floor(normalized);
      float p = normalized - floor;
      int bernoulli = static_cast<float>(_rand_list[i]) < p * MAX;
      unsigned quantized = floor + bernoulli;
      if (quantized) {
        size_t diff = i - last_non_zero_pos;
        last_non_zero_pos = i;
        EliasDeltaEncode(bit_writer, diff);
        int signbit = std::signbit(src[i]);
        bit_writer.Put(signbit);
        EliasDeltaEncode(bit_writer, quantized);
        error[i] -= (1 - (signbit << 1)) * quantized;
      }
    }
  } else if (_ptype == PartitionType::NATURAL) {
    const unsigned level = 1 << (_s - 1);
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * level;
      unsigned floor = RoundNextPow2(std::ceil(normalized)) >> 1;
      unsigned length = (floor != 0) ? floor : 1;
      float p = (normalized - floor) / length;
      int bernoulli = static_cast<float>(_rand_list[i]) < p * MAX;
      unsigned quantized = floor + length * bernoulli;
      if (quantized) {
        size_t diff = i - last_non_zero_pos;
        last_non_zero_pos = i;
        EliasDeltaEncode(bit_writer, diff);
        int signbit = std::signbit(src[i]);
        bit_writer.Put(signbit);
        EliasDeltaEncode(bit_writer, quantized);
        error[i] -= (1 - (signbit << 1)) * quantized;
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

  return bit_writer.blocks() * sizeof(index_t) + sizeof(index_t) +
         sizeof(float);
}

template <typename index_t, typename scalar_t>
size_t DitheringCompressor::FusedCompressImpl(index_t* __restrict__ dst,
                                              const scalar_t* __restrict__ src,
                                              scalar_t* __restrict__ error,
                                              size_t len) {
  _rand_list.clear();
  for (size_t i = 0; i < len; ++i) {
    _rand_list.push_back(_rng.xorshift128p());
  }
  if (std::is_same<index_t, int8_t>::value ||
      std::is_same<index_t, int16_t>::value) {
    return FusedCompressImplMax<index_t, scalar_t>(dst, src, error, len);
  } else {
    return FusedCompressImplL2<index_t, scalar_t>(dst, src, error, len);
  }
}
template <typename index_t, typename scalar_t>
size_t DitheringCompressor::FusedCompressImplMax(
    index_t* __restrict__ dst, const scalar_t* __restrict__ src,
    scalar_t* __restrict__ error, size_t len) {
  double scale = 0.0;
#pragma omp parallel for simd reduction(max : scale)
  for (size_t i = 0; i < len; i++) {
    scale = scale > std::abs(src[i]) ? scale : std::abs(src[i]);
  }
  const uint64_t MAX = std::numeric_limits<uint64_t>::max();

  if (_ptype == PartitionType::LINEAR) {
#pragma omp parallel for simd
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * _s;
      float floor = std::floor(normalized);
      float p = normalized - floor;
      int bernoulli = static_cast<float>(_rand_list[i]) < p * MAX;
      index_t quantized = floor + bernoulli;
      dst[i] = sgn(src[i]) * quantized;
      error[i] = src[i] - dst[i];
    }
  } else if (_ptype == PartitionType::NATURAL) {
    const unsigned level = 1 << (_s - 1);
#pragma omp parallel for simd
    for (size_t i = 0; i < len; ++i) {
      float abs_x = std::abs(src[i]);
      float normalized = (abs_x / scale) * level;
      unsigned floor = RoundNextPow2(std::ceil(normalized)) >> 1;
      unsigned length = (floor != 0) ? floor : 1;
      float p = (normalized - floor) / length;
      int bernoulli = static_cast<float>(_rand_list[i]) < p * MAX;
      index_t quantized = floor + length * bernoulli;
      dst[i] = sgn(src[i]) * quantized;
      error[i] = src[i] - dst[i];
    }
  }

  auto ptr = reinterpret_cast<float*>(&dst[len]);
  *ptr = scale;

  return len * sizeof(index_t) + sizeof(float);
}

void DitheringCompressor::FusedCompress(tensor_t grad, tensor_t& output,
                                        tensor_t error) {
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.data != output.data);
  if (output.data == nullptr) {
    output.data = _buf.get();
  }

  size_t compressed_size;
  if (this->_ntype == NormalizeType::L2) {
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
  } else if (this->_ntype == NormalizeType::MAX) {
    if (this->_s < (1 << 7)) {
      switch (grad.dtype) {
        case BYTEPS_FLOAT16:
          compressed_size =
              FusedCompressImpl(reinterpret_cast<int8_t*>(output.data),
                                reinterpret_cast<const half_t*>(grad.data),
                                reinterpret_cast<half_t*>(error.data),
                                grad.size / sizeof(half_t));
          break;
        case BYTEPS_FLOAT32:
          compressed_size = FusedCompressImpl(
              reinterpret_cast<int8_t*>(output.data),
              reinterpret_cast<const float*>(grad.data),
              reinterpret_cast<float*>(error.data), grad.size / sizeof(float));
          break;
        case BYTEPS_FLOAT64:
          compressed_size =
              FusedCompressImpl(reinterpret_cast<int8_t*>(output.data),
                                reinterpret_cast<const double*>(grad.data),
                                reinterpret_cast<double*>(error.data),
                                grad.size / sizeof(double));
          break;
        default:
          BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
      }
    } else if (this->_s < (1 << 15)) {
      switch (grad.dtype) {
        case BYTEPS_FLOAT16:
          compressed_size =
              FusedCompressImpl(reinterpret_cast<int16_t*>(output.data),
                                reinterpret_cast<const half_t*>(grad.data),
                                reinterpret_cast<half_t*>(error.data),
                                grad.size / sizeof(half_t));
          break;
        case BYTEPS_FLOAT32:
          compressed_size = FusedCompressImpl(
              reinterpret_cast<int16_t*>(output.data),
              reinterpret_cast<const float*>(grad.data),
              reinterpret_cast<float*>(error.data), grad.size / sizeof(float));
          break;
        case BYTEPS_FLOAT64:
          compressed_size =
              FusedCompressImpl(reinterpret_cast<int16_t*>(output.data),
                                reinterpret_cast<const double*>(grad.data),
                                reinterpret_cast<double*>(error.data),
                                grad.size / sizeof(double));
          break;
        default:
          BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
      }
    } else {
      BPS_CHECK(0) << "k exceeds the maximum limit.";
    }
  } else {
    BPS_CHECK(0) << "unsupport ntype";
  }

  output.size = compressed_size;
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps