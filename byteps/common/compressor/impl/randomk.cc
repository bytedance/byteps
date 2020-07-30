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

#include "../compressor_registry.h"
#include "randomk.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "randomk_compressor",
    [](const kwargs_t& kwargs, size_t size,
       DataType dtype) -> std::unique_ptr<Compressor> {
      auto factor = HyperParamFinder<float>(kwargs, "compressor_k", false,
                                            [](float x) { return x > 0; });
      unsigned k;
      if (factor < 1) {
        k = static_cast<unsigned>(factor * size / getDataTypeLength(dtype));
        if (k == 0) k = 1;
      } else {
        k = static_cast<unsigned>(factor);
      }

      auto seed = HyperParamFinder<unsigned>(kwargs, "seed", true,
                                             [](unsigned x) { return x != 0; });

      return std::unique_ptr<Compressor>(
          new RandomkCompressor(size, dtype, k, seed));
    });
}

template <typename index_t, typename scalar_t>
tensor_t RandomkCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                         size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  BPS_CHECK_LE(this->_k, len / 2);
  using pair_t = std::pair<index_t, scalar_t>;
  auto ptr = reinterpret_cast<pair_t*>(dst);

  for (size_t i = 0; i < this->_k; ++i) {
    auto index = _rng.Randint(0, len);
    ptr[i] = std::make_pair(index, src[index]);
  }

  return {dst, this->_k * sizeof(pair_t)};
}

tensor_t RandomkCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename index_t, typename scalar_t>
tensor_t RandomkCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                           size_t compressed_size) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  using pair_t = std::pair<index_t, scalar_t>;

  auto ptr = reinterpret_cast<const pair_t*>(src);
  if ((void*)dst == (void*)src) {
    auto buf = reinterpret_cast<pair_t*>(_buf.get());
    std::memcpy(buf, ptr, compressed_size);
    ptr = const_cast<const pair_t*>(buf);
  }

  // reset to zeros
  std::memset(dst, 0, _size);
  size_t len = compressed_size / sizeof(pair_t);
  for (size_t i = 0; i < len; ++i) {
    auto& pair = ptr[i];
    dst[pair.first] = pair.second;
  }

  return {dst, _size};
}

tensor_t RandomkCompressor::Decompress(tensor_t compressed) {
#ifdef BYTEPS_BUILDING_SERVER
  auto dst = _buf.get();
#else
  auto dst = compressed.data;
#endif
  DECOMPRESS_IMPL_SWITCH(_dtype, DecompressImpl, dst, compressed.data,
                         compressed.size);
}

template <typename index_t, typename scalar_t>
void RandomkCompressor::FastUpdateErrorImpl(scalar_t* error,
                                            scalar_t* corrected,
                                            const index_t* compressed,
                                            size_t compressed_size) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  using pair_t = std::pair<index_t, scalar_t>;

  std::memcpy(error, corrected, _size);

  auto ptr = reinterpret_cast<const pair_t*>(compressed);
  for (size_t i = 0; i < this->_k; ++i) {
    auto& pair = ptr[i];
    error[pair.first] = 0;
  }
}

void RandomkCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                        tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(_dtype, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps