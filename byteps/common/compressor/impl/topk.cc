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

#include <cstring>
#include <queue>

#include "../compressor_registry.h"
#include "topk.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "topk_compressor",
    [](const kwargs_t& kwargs, size_t size,
       DataType dtype) -> std::unique_ptr<Compressor> {
      auto compress_ratio = HyperParamFinder<float>(kwargs, "compressor_k", false,
                                            [](float x) { return x > 0; });
      return std::unique_ptr<Compressor>(
        new TopkCompressor(size, dtype, size/sizeof(float), compress_ratio));
    });
}

template <typename index_t, typename scalar_t>
tensor_t TopkCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                      size_t numel) {
  // no matter src is FP16 or FP32, the value is FP32 after compression
  using pair_t = std::pair<int, float>;
  auto comp = [](const pair_t& lhs, const pair_t& rhs) {
    return std::abs(lhs.second) > std::abs(rhs.second);
  };

  auto beg = reinterpret_cast<pair_t*>(dst);
  size_t k = std::ceil(_numel * _compress_ratio);
  size_t size = 0;
  for (size_t i = 0; i < numel; ++i) {
    if (i < k) {
      beg[size] = std::make_pair(float(i), src[i]);
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

  return {dst, k * sizeof(pair_t)};
}

tensor_t TopkCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename index_t, typename scalar_t>
tensor_t TopkCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                        size_t compressed_size) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  using pair_t = std::pair<int, float>;

  auto ptr = reinterpret_cast<const pair_t*>(src);
  // reset to zeros
  std::memset(dst, 0, _numel*4);
  size_t numel = compressed_size / sizeof(pair_t);

#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < numel; ++i) {
    auto& pair = ptr[i];
    dst[pair.first] = pair.second;
  }

  return {dst, _numel * sizeof(float)};
}

tensor_t TopkCompressor::Decompress(tensor_t compressed) {
  DECOMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, DecompressImpl, _buf.get(), compressed.data,
                         compressed.size);
}

template <typename index_t, typename scalar_t>
void TopkCompressor::FastUpdateErrorImpl(scalar_t* error, scalar_t* corrected,
                                         const index_t* compressed,
                                         size_t compressed_size) {
  using pair_t = std::pair<int, float>;
  size_t k = _numel * _compress_ratio;
  std::memcpy(error, corrected, _size);
  auto ptr = reinterpret_cast<const pair_t*>(compressed);
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < k; ++i) {
    auto& pair = ptr[i];
    error[pair.first] = 0;
  }
}

void TopkCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                     tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(BYTEPS_FLOAT32, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps