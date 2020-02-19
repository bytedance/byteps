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
  return std::unique_ptr<BaseCompressor>(new OnebitCompressor());
});
}

OnebitCompressor::OnebitCompressor() = default;

OnebitCompressor::~OnebitCompressor() = default;

size_t Packing(void* src, size_t len) {
  constexpr int MASK = 1;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  size_t chunk_size = (len + padding_len) / PACKING_SIZE;

  auto ptr_src = reinterpret_cast<int*>(src);
#pragma unroll
  for (int i = 1; i < PACKING_SIZE; ++i) {
    for (int j = 0; j < chunk_size; ++j) {
      ptr_src[j] <<= 1;
      ptr_src[j] |= ptr_src[i * chunk_size + j] & MASK;
    }
  }

  return chunk_size;
}

ByteBuf OnebitCompressor::Compress(const ByteBuf& grad) {
  // BPS_CHECK_EQ(grad.len, _src_len);
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.len);
  BPS_CHECK(_encode_buf);

  auto pos_0 = std::chrono::steady_clock::now();
  auto reduced_len = _cpu_reducer->sign(_encode_buf.get(), grad.data, grad.len,
                                        static_cast<DataType>(grad.dtype));

  auto pos_1 = std::chrono::steady_clock::now();

  auto compressed_len = Packing(_encode_buf.get(), reduced_len);

  auto pos_2 = std::chrono::steady_clock::now();

  auto duration_0 =
      std::chrono::duration_cast<std::chrono::microseconds>(pos_1 - pos_0);
  auto duration_1 =
      std::chrono::duration_cast<std::chrono::microseconds>(pos_2 - pos_1);

  double elapsed_0 = double(duration_0.count());
  double elapsed_1 = double(duration_1.count());

  BPS_LOG(INFO) << "Time elapsed for compress size=" << grad.len
                << ", sign=" << elapsed_0 << "ms"
                << ", packing=" << elapsed_1 << "ms";

  return {_encode_buf.get(), compressed_len, grad.dtype};
}

void Unpacking(void* dst, void* src, size_t len) {
  constexpr int MASK = 1;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  size_t chunk_size = (len + padding_len) / PACKING_SIZE;

  auto ptr_dst = reinterpret_cast<int*>(dst);
  auto ptr_src = reinterpret_cast<int*>(src);
#pragma unroll
  for (int i = PACKING_SIZE - 1; i >= 0; --i) {
    for (int j = 0; j < chunk_size; ++j) {
      ptr_dst[i * chunk_size + j] = ~(ptr_src[j] & MASK) + 1;
      ptr_src[j] >>= 1;
    }
  }
}

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
ByteBuf OnebitCompressor::Decompress(const ByteBuf& compressed) {
  BPS_CHECK(compressed.data);
  BPS_CHECK(compressed.len);

  auto pos_0 = std::chrono::system_clock::now();

  Unpacking(_encode_buf.get(), compressed.data,
            _src_len / getDataTypeLength(compressed.dtype));

  auto pos_1 = std::chrono::system_clock::now();

  _cpu_reducer->int2fp(compressed.data, _encode_buf.get(), _src_len,
                       static_cast<DataType>(compressed.dtype));

  auto pos_2 = std::chrono::system_clock::now();

  auto duration_0 =
      std::chrono::duration_cast<std::chrono::microseconds>(pos_1 - pos_0);
  auto duration_1 =
      std::chrono::duration_cast<std::chrono::microseconds>(pos_2 - pos_1);

  double elapsed_0 = double(duration_0.count());
  double elapsed_1 = double(duration_1.count());

  BPS_LOG(INFO) << "Time elapsed for decompress src_size=" << _src_len
                << ", unpacking=" << elapsed_0 << "ms"
                << ", byte2float=" << elapsed_1 << "ms";

  return {nullptr, _src_len, 0};
}

#else

ByteBuf OnebitCompressor::Decompress(const ByteBuf& compressed) {
  BPS_CHECK(compressed.data);
  BPS_CHECK(compressed.len);

  Unpacking(_encode_buf.get(), compressed.data,
            _src_len / getDataTypeLength(compressed.dtype));
  _cpu_reducer->int2fp(_encode_buf.get(), _encode_buf.get(), _src_len,
                       static_cast<DataType>(compressed.dtype));
  return {_encode_buf.get(), _src_len, compressed.dtype};
}
#endif

}  // namespace compressor
}  // namespace common
}  // namespace byteps