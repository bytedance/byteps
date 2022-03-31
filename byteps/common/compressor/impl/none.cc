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
#include "none.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "none_compressor",
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
          new NoneCompressor(size, dtype, k, seed));
    });
}

template <typename index_t, typename scalar_t>
tensor_t NoneCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                         size_t len) {
  size_t num = len / sizeof(scalar_t);
  auto ptr = reinterpret_cast<scalar_t*>(dst);
  
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < num; ++i) {
    ptr[i] = src[i];
  }
  
  return {dst, len};
}

tensor_t NoneCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename index_t, typename scalar_t>
tensor_t NoneCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                           size_t compressed_size) {
  size_t num = compressed_size / sizeof(index_t);
  auto ptr = reinterpret_cast<index_t*>(dst);
  
#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < num; ++i) {
    ptr[i] = src[i];
  }
  
  return {dst, compressed_size};
}

tensor_t NoneCompressor::Decompress(tensor_t compressed) {
  std::cout << "NoneCompressor Decompress begin" << std::endl;
#ifdef BYTEPS_BUILDING_SERVER
  auto dst = _buf.get();
#else
  auto dst = compressed.data;
#endif
  std::cout << "NoneCompressor Decompress" << std::endl;
  DECOMPRESS_IMPL_SWITCH(_dtype, DecompressImpl, dst, compressed.data,
                         compressed.size);
}


template <typename index_t, typename scalar_t>
void NoneCompressor::FastUpdateErrorImpl(scalar_t* error,
                                            scalar_t* corrected,
                                            const index_t* compressed,
                                            size_t compressed_size) {
  return;
}

void NoneCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                        tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(_dtype, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps