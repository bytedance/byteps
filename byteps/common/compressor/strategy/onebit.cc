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

#include "../../global.h"
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

ByteBuf OnebitCompressor::Compress(const ByteBuf& grad) {
  BPS_CHECK(_encode_buf);
  BPS_CHECK_EQ(grad.len, _src_len);
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.len);
  auto reducer = BytePSGlobal::GetCpuReducer();
  auto reduced_size =
      reducer->sign(_encode_buf.get(), grad.data, grad.len, grad.dtype);
  auto compressed_size = Packing(_encode_buf.get(), reduced_size);
  return {_encode_buf.get(), compressed_size, grad.dtype};
}

ByteBuf OnebitCompressor::Decompress(const ByteBuf& compressed) {
  // TODO
}

size_t OnebitCompressor::Packing(char* data, size_t len) {
  size_t padding_len = (BYTE_SIZE - (len % BYTE_SIZE)) % BYTE_SIZE;
  size_t total_len = len + padding_len;
  size_t total_bytes = total_len / BYTE_SIZE;
  const char mask = 1;
  for (int i = 0, base = 0; i < total_bytes; ++i, base += BYTE_SIZE) {
    data[i] |= (data[base] & mask);
    for (int j = 1; j < BYTE_SIZE; ++j) {
      data[i] <<= 1;
      if (base + j < len) {
        data[i] |= (data[base + j] & mask);
      }
    }
  }

  return total_bytes;
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps