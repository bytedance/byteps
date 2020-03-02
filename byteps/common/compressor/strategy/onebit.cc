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

size_t Packing(void* data, size_t len, float scale) {
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  size_t chunk_size = (len + padding_len) / PACKING_SIZE;

  auto ptr = reinterpret_cast<int*>(data);
#pragma unroll
  for (int i = 1; i < PACKING_SIZE; ++i) {
    for (int j = 0; j < chunk_size; ++j) {
      ptr[j] <<= 1;
      ptr[j] |= ptr[i * chunk_size + j] & 0x01;
    }
  }
  auto ptr_fp = reinterpret_cast<float*>(data);
  ptr_fp[chunk_size] = scale;

  return chunk_size + 1;
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

  auto compressed_len = Packing(_buf.get(), reduced_len, scale);

  compressed.data = _buf.get();
  compressed.size = compressed_len * sizeof(int);
}

void Unpacking(void* dst, const void* src, size_t size, float* scale) {
  BPS_CHECK_NE(dst, src);
  auto chunk_size = size / sizeof(int) - 1;

  auto ptr_dst = reinterpret_cast<int*>(dst);
  auto ptr_src = reinterpret_cast<const int*>(src);
  unsigned int mask = 1;
#pragma unroll
  for (int i = PACKING_SIZE - 1; i >= 0; --i) {
    for (int j = 0; j < chunk_size; ++j) {
      int sign_bit = (ptr_src[j] & mask) >> (PACKING_SIZE - i - 1);
      ptr_dst[i * chunk_size + j] = -((sign_bit << 1) - 1);
    }
    mask <<= 1;
  }

  auto ptr_src_fp = reinterpret_cast<const float*>(src);
  *scale = ptr_src_fp[chunk_size];
}

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
void OnebitCompressor::Decompress(ByteBuf compressed, int dtype,
                                  ByteBuf& decompressed) {
  BPS_CHECK(decompressed.data);
  float scale;
  // core_loop's
  if (decompressed.data == compressed.data) {
    Unpacking(_buf.get(), compressed.data, compressed.size, &scale);
    _cpu_reducer->int2fp(decompressed.data, _buf.get(), decompressed.size,
                         static_cast<DataType>(dtype), scale);
  } else {
    // error feedback
    Unpacking(decompressed.data, compressed.data, compressed.size, &scale);
    _cpu_reducer->int2fp(decompressed.data, decompressed.data,
                         decompressed.size, static_cast<DataType>(dtype),
                         scale);
  }
}

#else

void OnebitCompressor::Decompress(ByteBuf compressed, int dtype,
                                  ByteBuf& decompressed) {
  BPS_CHECK(decompressed);
  float scale;
  if (decompressed.data == nullptr) decompressed.data = _buf.get();
  Unpacking(decompressed.data, compressed.data, compressed.size, &scale);
  _cpu_reducer->int2fp(decompressed.data, decompressed.data,
                       decompressed.size, static_cast<DataType>(dtype), scale);
}
#endif

}  // namespace compressor
}  // namespace common
}  // namespace byteps