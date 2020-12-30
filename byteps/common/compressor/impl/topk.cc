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

#include <cstring>
#include <queue>
#include <sstream>

#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "topk_compressor",
    [](const kwargs_t& kwargs, size_t size, DataType dtype,
       std::unique_ptr<Compressor> cptr) -> std::unique_ptr<Compressor> {
      auto factor = HyperParamFinder<float>(kwargs, "compressor_k", false,
                                            [](float x) { return x > 0; });
      unsigned k;
      if (factor < 1) {
        k = static_cast<unsigned>(factor * size / getDataTypeLength(dtype));
        if (k == 0) k = 1;
      } else {
        k = static_cast<unsigned>(factor);
      }

      BPS_LOG(INFO) << "topk compressor is registered."
                    << "\tsize=" << size << "\tk=" << k;
      return std::unique_ptr<Compressor>(new TopkCompressor(size, dtype, k));
    });
}

template <typename pair_t, typename scalar_t>
size_t TopkCompressor::CompressImpl(pair_t* __restrict__ dst,
                                    const scalar_t* __restrict__ src,
                                    size_t len) {
  BPS_CHECK_LE(this->_k, len / 2);
  auto comp = [](const pair_t& lhs, const pair_t& rhs) {
    return std::abs(lhs.second) > std::abs(rhs.second);
  };

  auto beg = reinterpret_cast<pair_t*>(dst);
  size_t size = 0;
  for (size_t i = 0; i < len; ++i) {
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

  return this->_k * sizeof(pair_t);
}  // namespace compressor

void TopkCompressor::Compress(tensor_t grad, tensor_t& output) {
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.data != output.data);
  if (output.data == nullptr) {
    output.data = _buf.get();
  }

  size_t compressed_size;
  switch (grad.dtype) {
    case BYTEPS_FLOAT16:
      compressed_size = CompressImpl(
          reinterpret_cast<std::pair<uint32_t, float>*>(output.data),
          reinterpret_cast<const half_t*>(grad.data),
          grad.size / sizeof(half_t));
      break;
    case BYTEPS_FLOAT32:
      compressed_size = CompressImpl(
          reinterpret_cast<std::pair<uint32_t, float>*>(output.data),
          reinterpret_cast<const float*>(grad.data), grad.size / sizeof(float));
      break;
    case BYTEPS_FLOAT64:
      compressed_size = CompressImpl(
          reinterpret_cast<std::pair<uint64_t, double>*>(output.data),
          reinterpret_cast<const double*>(grad.data),
          grad.size / sizeof(double));
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
  }

  output.size = compressed_size;
}

template <typename scalar_t, typename pair_t>
void TopkCompressor::DecompressImpl(scalar_t* __restrict__ dst,
                                    const pair_t* __restrict__ src,
                                    size_t compressed_size, size_t dst_size) {
  auto ptr = reinterpret_cast<const pair_t*>(src);
  // reset to zeros
  std::memset(dst, 0, dst_size);
  size_t len = compressed_size / sizeof(pair_t);
  BPS_LOG(INFO) << "max=" << ptr[0].second << " min=" << ptr[len - 1].second
                << " k=" << len;
  for (size_t i = 0; i < len; ++i) {
    auto& pair = ptr[i];
    dst[pair.first] = pair.second;
  }
}

void TopkCompressor::Decompress(tensor_t compressed, tensor_t& output) {
  BPS_CHECK(compressed.data);
  BPS_CHECK(compressed.data != output.data);

  if (output.data == nullptr) {
    output = {_buf.get(), _size, _dtype};
  } else {
    BPS_CHECK(output.size > 0);
  }

  switch (output.dtype) {
    case BYTEPS_FLOAT16:
      DecompressImpl(
          reinterpret_cast<half_t*>(output.data),
          reinterpret_cast<const std::pair<uint32_t, float>*>(compressed.data),
          compressed.size, output.size);
      break;
    case BYTEPS_FLOAT32:
      DecompressImpl(
          reinterpret_cast<float*>(output.data),
          reinterpret_cast<const std::pair<uint32_t, float>*>(compressed.data),
          compressed.size, output.size);
      break;
    case BYTEPS_FLOAT64:
      DecompressImpl(
          reinterpret_cast<double*>(output.data),
          reinterpret_cast<const std::pair<uint64_t, double>*>(compressed.data),
          compressed.size, output.size);
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << output.dtype;
  }
}

template <typename pair_t, typename scalar_t>
size_t TopkCompressor::FusedCompressImpl(pair_t* __restrict__ dst,
                                         const scalar_t* __restrict__ src,
                                         scalar_t* __restrict__ error,
                                         size_t len) {
  BPS_CHECK_LE(this->_k, len / 2);
  auto comp = [](const pair_t& lhs, const pair_t& rhs) {
    return std::abs(lhs.second) > std::abs(rhs.second);
  };

  std::memcpy(error, src, len * sizeof(scalar_t));

  auto beg = reinterpret_cast<pair_t*>(dst);
  size_t size = 0;
  for (size_t i = 0; i < len; ++i) {
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

  for (int i = 0; i < this->_k; ++i) {
    error[beg[i].first] = 0;
  }

  return this->_k * sizeof(pair_t);
}

void TopkCompressor::FusedCompress(tensor_t grad, tensor_t& output,
                                   tensor_t error) {
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.data != output.data);
  if (output.data == nullptr) {
    output.data = _buf.get();
  }

  size_t compressed_size;
  switch (grad.dtype) {
    case BYTEPS_FLOAT16:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<std::pair<uint32_t, float>*>(output.data),
          reinterpret_cast<const half_t*>(grad.data),
          reinterpret_cast<half_t*>(error.data), grad.size / sizeof(half_t));
      break;
    case BYTEPS_FLOAT32:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<std::pair<uint32_t, float>*>(output.data),
          reinterpret_cast<const float*>(grad.data),
          reinterpret_cast<float*>(error.data), grad.size / sizeof(float));
      break;
    case BYTEPS_FLOAT64:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<std::pair<uint64_t, double>*>(output.data),
          reinterpret_cast<const double*>(grad.data),
          reinterpret_cast<double*>(error.data), grad.size / sizeof(double));
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
  }

  output.size = compressed_size;
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps