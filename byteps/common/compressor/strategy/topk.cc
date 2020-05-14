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

#include "topk.h"

#include <queue>

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "topk_compressor",
    [](const kwargs_t& kwargs) -> std::unique_ptr<BaseCompressor> {
      auto iter = kwargs.find("compressor_k");
      if (iter == kwargs.end()) {
        BPS_LOG(WARNING) << "Topk Compressor needs parameter \"compressor_k\"";
        return nullptr;
      }
      int k = std::stoi(iter->second);
      BPS_LOG(DEBUG) << "Register Topk Compressor "
                     << "k=" << k;
      return std::unique_ptr<BaseCompressor>(new TopkCompressor(k));
    });
}

TopkCompressor::TopkCompressor(int k) : _k(k){};

TopkCompressor::~TopkCompressor() = default;

template <typename index_t, typename scalar_t>
size_t TopkCompressor::_Packing(index_t* dst, const scalar_t* src, size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  BPS_CHECK_LE(this->_k, len / 2);
  using pair_t = std::pair<index_t, scalar_t>;
  using container_t = std::vector<pair_t>;
  auto comp = [](const pair_t& lhs, const pair_t& rhs) {
    return lhs.second > rhs.second;
  };
  this->_src_len = len;
  auto beg = reinterpret_cast<pair_t*>(dst);
  size_t size = 0;
  for (index_t i = 0; i < len; ++i) {
    if (i < this->_k) {
      beg[size] = std::make_pair(i, src[i]);
      size++;
      std::push_heap(beg, beg + size, comp);
    } else {
      auto& top = *beg;
      // note: compare absolute value
      if (std::abs(src[i]) > std::abs(top.second)) {
        std::pop_heap(beg, beg + size, comp);
        beg[size - 1] = std::make_pair(i, src[i]);
        std::push_heap(beg, beg + size, comp);
      }
    }
  }
  BPS_LOG(INFO) << "first=" << beg[0].first << " second=" << beg[0].second;

  return this->_k * sizeof(pair_t);
}

size_t TopkCompressor::Packing(const void* src, size_t size, int dtype) {
  switch (dtype) {
    case BYTEPS_INT8:
      return _Packing(reinterpret_cast<int8_t*>(_buf.get()),
                      reinterpret_cast<const int8_t*>(src),
                      size / sizeof(int8_t));
    case BYTEPS_UINT8:
      return _Packing(reinterpret_cast<uint8_t*>(_buf.get()),
                      reinterpret_cast<const uint8_t*>(src),
                      size / sizeof(uint8_t));
    // case BYTEPS_FLOAT16:
    //   return _Packing(reinterpret_cast<int8_t*>(_buf.get()),
    //                   reinterpret_cast<const int8_t*>(src), size);
    case BYTEPS_FLOAT32:
      return _Packing(reinterpret_cast<int32_t*>(_buf.get()),
                      reinterpret_cast<const float*>(src),
                      size / sizeof(int32_t));
    case BYTEPS_FLOAT64:
      return _Packing(reinterpret_cast<int64_t*>(_buf.get()),
                      reinterpret_cast<const double*>(src),
                      size / sizeof(int64_t));
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

void TopkCompressor::Compress(ByteBuf grad, int dtype, ByteBuf& compressed) {
  compressed.size = Packing(grad.data, grad.size, dtype);
  compressed.data = _buf.get();
}

template <typename index_t, typename scalar_t>
size_t TopkCompressor::_Unpacking(scalar_t* dst, const index_t* src,
                                  size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  using pair_t = std::pair<index_t, scalar_t>;
  auto ptr = reinterpret_cast<const pair_t*>(src);
  
  if ((void*)dst == (void*)src) {
    auto buf = reinterpret_cast<pair_t*>(_buf.get());
    std::copy(ptr, ptr+len, buf);
    ptr = const_cast<const pair_t*>(buf);
  }

  // reset to zeros
  std::fill(dst, dst + this->_src_len, 0);
  for (auto i = 0; i < len; ++i) {
    auto& pair = ptr[i];
    dst[pair.first] = pair.second;
  }
}

size_t TopkCompressor::Unpacking(void* dst, const void* src, size_t size,
                                 int dtype) {
  switch (dtype) {
    case BYTEPS_INT8:
      return _Unpacking(reinterpret_cast<int8_t*>(dst),
                        reinterpret_cast<const int8_t*>(src),
                        size / sizeof(int8_t) / 2);
    case BYTEPS_UINT8:
      return _Unpacking(reinterpret_cast<uint8_t*>(dst),
                        reinterpret_cast<const uint8_t*>(src),
                        size / sizeof(uint8_t) / 2);
    // case BYTEPS_FLOAT16:
    //   return _Unpacking(reinterpret_cast<int8_t*>(_buf.get()),
    //                   reinterpret_cast<const int8_t*>(src), size);
    case BYTEPS_FLOAT32:
      return _Unpacking(reinterpret_cast<float*>(dst),
                        reinterpret_cast<const int32_t*>(src),
                        size / sizeof(float) / 2);
    case BYTEPS_FLOAT64:
      return _Unpacking(reinterpret_cast<double*>(dst),
                        reinterpret_cast<const int64_t*>(src),
                        size / sizeof(double) / 2);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
void TopkCompressor::Decompress(ByteBuf compressed, int dtype,
                                ByteBuf& decompressed) {
  BPS_CHECK(decompressed.data);
  Unpacking(decompressed.data, compressed.data, compressed.size, dtype);
}
#else
void TopkCompressor::Decompress(ByteBuf compressed, int dtype,
                                ByteBuf& decompressed) {
  if (decompressed.data == nullptr) decompressed.data = _buf.get();
  Unpacking(decompressed.data, compressed.data, compressed.size, dtype);
}
#endif
}  // namespace compressor
}  // namespace common
}  // namespace byteps