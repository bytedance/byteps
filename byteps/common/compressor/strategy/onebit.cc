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

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg("onebit", [](const kwargs_t& kwargs) {
  BPS_LOG(DEBUG) << "Register Onebit Compressor";
  return std::unique_ptr<BaseCompressor>(new OnebitCompressor());
});
}

OnebitCompressor::OnebitCompressor() = default;

OnebitCompressor::~OnebitCompressor() = default;

size_t Packing(char* data, size_t len) {
  size_t padding_len = (BYTE_SIZE - (len % BYTE_SIZE)) % BYTE_SIZE;
  size_t total_len = len + padding_len;
  size_t compressed_len = total_len / BYTE_SIZE;
  constexpr unsigned char MASK = 0x01;
  for (size_t i = 0, base = 0; i < compressed_len; ++i, base += BYTE_SIZE) {
    data[i] = (data[base] & MASK);
#pragma unroll
    for (size_t j = 1; j < BYTE_SIZE; ++j) {
      data[i] <<= 1;
      if (base + j < len) {
        data[i] |= (data[base + j] & MASK);
      }
    }
  }

  return compressed_len;
}

ByteBuf OnebitCompressor::Compress(const ByteBuf& grad) {
  BPS_CHECK_EQ(grad.len, _src_len);
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.len);
  BPS_CHECK(_encode_buf);
  auto reduced_len = _cpu_reducer->sign(_encode_buf.get(), grad.data, grad.len,
                                        static_cast<DataType>(grad.dtype));
  auto compressed_len = Packing(_encode_buf.get(), reduced_len);
  return {_encode_buf.get(), compressed_len, grad.dtype};
}

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
void Unpacking(char* data, size_t len, size_t src_len, size_t stride) {
  constexpr unsigned char MASK = 0x01;
  size_t pos;
  for (int i = len - 1, base = BYTE_SIZE * i; i >= 0; --i, base -= BYTE_SIZE) {
#pragma unroll
    for (int j = BYTE_SIZE - 1; j >= 1; --j) {
      pos = (base + j) * stride;
      if (pos < src_len) {
        data[pos] = (data[i] & MASK);
      }
      data[i] >>= 1;
    }
    data[base * stride] = (data[i] & MASK);
  }
}

ByteBuf OnebitCompressor::Decompress(const ByteBuf& compressed) {
  BPS_CHECK(compressed.data);
  BPS_CHECK(compressed.len);

  Unpacking(compressed.data, compressed.len, _src_len,
            getDataTypeLength(compressed.dtype));
  _cpu_reducer->byte2float(compressed.data, _src_len,
                           static_cast<DataType>(compressed.dtype));
  return {nullptr, _src_len, 0};
}

#else
// server version decompressor
void Unpacking(char* dst, char* src, size_t len, size_t src_len,
               size_t stride) {
  constexpr unsigned char MASK = 0x80;
  size_t pos;
  for (size_t i = 0, base = 0; i < len; ++i, base += BYTE_SIZE) {
#pragma unroll
    for (size_t j = 0; j < BYTE_SIZE; ++j) {
      pos = (base + j) * stride;
      if (pos >= src_len) {
        break;
      }
      dst[pos] = (src[i] & MASK);
      src[i] <<= 1;
    }
  }
}

ByteBuf OnebitCompressor::Decompress(const ByteBuf& compressed) {
  BPS_CHECK(compressed.data);
  BPS_CHECK(compressed.len);

  Unpacking(_encode_buf.get(), compressed.data, compressed.len, _src_len,
            getDataTypeLength(compressed.dtype));
  _cpu_reducer->byte2float(_encode_buf.get(), _src_len,
                           static_cast<DataType>(compressed.dtype));
  return {_encode_buf.get(), _src_len, compressed.dtype};
}
#endif

}  // namespace compressor
}  // namespace common
}  // namespace byteps