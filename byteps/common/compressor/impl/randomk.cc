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
#include <cmath>

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
      auto compress_ratio = HyperParamFinder<float>(kwargs, "compressor_k", false,
                                            [](float x) { return x > 0; });

      auto seed = HyperParamFinder<unsigned>(kwargs, "seed", true,
                                             [](unsigned x) { return x != 0; });
      return std::unique_ptr<Compressor>(
          new RandomkCompressor(size, dtype, size/sizeof(float), compress_ratio, seed));
    });
}


template <typename index_t, typename scalar_t>
tensor_t RandomkCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                         size_t numel) {
  
  // no matter src is FP16 or FP32, the value is FP32 after compression
  using pair_t = std::pair<int, float>;
  auto ptr = reinterpret_cast<pair_t*>(dst);
  size_t k = std::ceil(_numel * _compress_ratio);

#pragma omp parallel for simd
  for (size_t i = 0; i < k; ++i) {
    auto index = _rng.Randint(0, _numel);
    ptr[i] = std::make_pair(index, src[index]);
  }

  BPS_LOG(INFO) << "RandomKCompressor CompressImpl numel: " << this->_numel
  << ", len: " << numel 
  << ", compress_ratio: " << _compress_ratio
  << std::endl;

  return {dst, k * sizeof(pair_t)};
}


tensor_t RandomkCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}


template <typename index_t, typename scalar_t>
tensor_t RandomkCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                           size_t compressed_size) {
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

  BPS_LOG(INFO) << "RandomKCompressor after DecompressImpl numel: " << this->_numel
    << ", len: " << numel 
    << ", compressed_size: " << compressed_size
    << std::endl;

    return {dst, _numel * sizeof(float)};
}

tensor_t RandomkCompressor::Decompress(tensor_t compressed) {
  DECOMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, DecompressImpl, _buf.get(), compressed.data,
                         compressed.size);
}


template <typename index_t, typename scalar_t>
void RandomkCompressor::FastUpdateErrorImpl(scalar_t* error,
                                            scalar_t* corrected,
                                            const index_t* compressed,
                                            size_t compressed_size) {
  using pair_t = std::pair<int, float>;
  std::memcpy(error, corrected, _size);
  auto ptr = reinterpret_cast<const pair_t*>(compressed);
  size_t k = _numel * _compress_ratio;
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < k; ++i) {
    auto& pair = ptr[i];
    error[pair.first] = 0;
  }
}

void RandomkCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                        tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(BYTEPS_FLOAT32, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps