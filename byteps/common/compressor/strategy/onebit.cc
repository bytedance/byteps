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

#include <chrono>

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg("onebit", [](const kwargs_t& kwargs) {
  BPS_LOG(DEBUG) << "Register Onebit Compressor";
  if (kwargs.find("compressor_onebit_enable_scale") != kwargs.end()) {
    return std::unique_ptr<BaseCompressor>(new OnebitCompressor(true));
  }
  return std::unique_ptr<BaseCompressor>(new OnebitCompressor());
});
}

OnebitCompressor::OnebitCompressor(bool use_scale) : _use_scale(use_scale){};

OnebitCompressor::~OnebitCompressor() = default;

template <typename T>
size_t _Packing(T* data, size_t len, float scale) {
  constexpr int PACKING_SIZE = sizeof(T) * 8;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  size_t chunk_size = (len + padding_len) / PACKING_SIZE;

  for (int i = 1; i < PACKING_SIZE; ++i) {
    for (int j = 0; j < chunk_size; ++j) {
      data[j] <<= 1;
      data[j] |= data[i * chunk_size + j] & 0x01;
    }
  }
  auto ptr_fp = reinterpret_cast<float*>(data);
  ptr_fp[chunk_size] = scale;

  return chunk_size + 1;
}

size_t Packing(void* data, size_t len, float scale, int dtype) {
  switch (dtype) {
    case BYTEPS_INT8:
    case BYTEPS_UINT8:
      return _Packing(reinterpret_cast<int8_t*>(data), len, scale);
    case BYTEPS_FLOAT16:
      return _Packing(reinterpret_cast<int16_t*>(data), len, scale);
    case BYTEPS_INT32:
    case BYTEPS_FLOAT32:
      return _Packing(reinterpret_cast<int32_t*>(data), len, scale);
    case BYTEPS_INT64:
    case BYTEPS_FLOAT64:
      return _Packing(reinterpret_cast<int64_t*>(data), len, scale);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

void OnebitCompressor::Compress(ByteBuf grad, int dtype, ByteBuf& compressed) {
  float scale = 1.0;
  if (_use_scale) {
    float norm1;
    _cpu_reducer->norm1(&norm1, grad.data, grad.size,
                        static_cast<DataType>(dtype));
    scale = norm1 / (grad.size / getDataTypeLength(dtype));
  }

  auto reduced_len = _cpu_reducer->sign(_buf.get(), grad.data, grad.size,
                                        static_cast<DataType>(dtype));

  auto compressed_len = Packing(_buf.get(), reduced_len, scale, dtype);

  compressed.data = _buf.get();
  compressed.size = compressed_len * sizeof(int);
}

template <typename T>
size_t _Unpacking(T* dst, const T* src, size_t size, float* scale) {
  constexpr int PACKING_SIZE = sizeof(T) * 8;
  auto chunk_size = size / sizeof(int) - 1;

  unsigned int mask = 1;
  for (int i = PACKING_SIZE - 1; i >= 0; --i) {
    for (int j = 0; j < chunk_size; ++j) {
      int sign_bit = (src[j] & mask) >> (PACKING_SIZE - i - 1);
      dst[i * chunk_size + j] = -((sign_bit << 1) - 1);
    }
    mask <<= 1;
  }

  auto ptr_src_fp = reinterpret_cast<const float*>(src);
  *scale = ptr_src_fp[chunk_size];
  return chunk_size;
}

size_t Unpacking(void* dst, const void* src, size_t len, float* scale,
                 int dtype) {
  BPS_CHECK_NE(dst, src);
  switch (dtype) {
    case BYTEPS_INT8:
    case BYTEPS_UINT8:
      return _Unpacking(reinterpret_cast<int8_t*>(dst),
                        reinterpret_cast<const int8_t*>(src), len, scale);
    case BYTEPS_FLOAT16:
      return _Unpacking(reinterpret_cast<int16_t*>(dst),
                        reinterpret_cast<const int16_t*>(src), len, scale);
    case BYTEPS_INT32:
    case BYTEPS_FLOAT32:
      return _Unpacking(reinterpret_cast<int32_t*>(dst),
                        reinterpret_cast<const int32_t*>(src), len, scale);
    case BYTEPS_INT64:
    case BYTEPS_FLOAT64:
      return _Unpacking(reinterpret_cast<int64_t*>(dst),
                        reinterpret_cast<const int64_t*>(src), len, scale);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
void OnebitCompressor::Decompress(ByteBuf compressed, int dtype,
                                  ByteBuf& decompressed) {
  BPS_CHECK(decompressed.data);
  float scale;
  // core_loop's
  if (decompressed.data == compressed.data) {
    Unpacking(_buf.get(), compressed.data, compressed.size, &scale, dtype);
    _cpu_reducer->scale(decompressed.data, _buf.get(), decompressed.size,
                        static_cast<DataType>(dtype), scale);
  } else {
    // error feedback
    Unpacking(decompressed.data, compressed.data, compressed.size, &scale,
              dtype);
    _cpu_reducer->scale(decompressed.data, decompressed.data, decompressed.size,
                        static_cast<DataType>(dtype), scale);
  }
}

#else

void OnebitCompressor::Decompress(ByteBuf compressed, int dtype,
                                  ByteBuf& decompressed) {
  BPS_CHECK(decompressed.data);
  float scale;
  if (decompressed.data == nullptr) decompressed.data = _buf.get();
  Unpacking(decompressed.data, compressed.data, compressed.size, &scale, dtype);
  _cpu_reducer->scale(decompressed.data, decompressed.data, decompressed.size,
                      static_cast<DataType>(dtype), scale);
}
#endif

}  // namespace compressor
}  // namespace common
}  // namespace byteps