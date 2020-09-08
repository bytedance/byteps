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

#include "randomk.h"

#include <cstring>

#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "randomk_compressor",
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

      auto seed = HyperParamFinder<unsigned>(kwargs, "seed", true,
                                             [](unsigned x) { return x != 0; });

      bool is_scale = false;
      // ef not enabled
      if (kwargs.find("ef_type") == kwargs.end()) {
        is_scale = true;
      }

      BPS_LOG(INFO) << "randomk compressor is registered. "
                    << "\tsize=" << size << "\tk=" << k << "\tseed=" << seed
                    << "\tis_scale=" << is_scale;

      return std::unique_ptr<Compressor>(
          new RandomkCompressor(size, dtype, k, seed, is_scale));
    });
}

template <typename scalar_t>
tensor_t RandomkCompressor::CompressImpl(scalar_t* dst, const scalar_t* src,
                                         size_t len) {
#ifndef BYTEPS_BUILDING_SERVER
  // generate non-overlap k
  while (_selected_set.size() < this->_k) {
    _selected_set.insert(_rng.Randint(0, len));
  }
  std::copy(_selected_set.begin(), _selected_set.end(),
            std::bacK_inserter(_selected_idx));
  _selected_set.clear();

  // to be unbiased
  if (_is_scale) {
    float scale = static_cast<float>(len) / this->_k;
#pragma omp parallel for simd
    for (size_t i = 0; i < this->_k; ++i) {
      dst[i] = src[_selected_idx[i]] * scale;
    }
  } else {
#pragma omp parallel for simd
    for (size_t i = 0; i < this->_k; ++i) {
      dst[i] = src[_selected_idx[i]];
    }
  }
#endif
  // server does nothing
  return {dst, this->_k * sizeof(scalar_t)};
}

tensor_t RandomkCompressor::Compress(tensor_t grad) {
#ifndef BYTEPS_BUILDING_SERVER
  auto dst = _buf.get();
#else
  auto dst = grad.data;
#endif
  COMPRESS_IMPL_SWITCH2(grad.dtype, CompressImpl, dst, grad.data, grad.size);
}

template <typename scalar_t>
tensor_t RandomkCompressor::DecompressImpl(scalar_t* dst, const scalar_t* src,
                                           size_t compressed_size) {
#ifndef BYTEPS_BUILDING_SERVER
  auto buf = reinterpret_cast<scalar_t*>(_buf.get());
  if ((void*)dst == (void*)src) {
    std::memcpy(buf, src, compressed_size);
  }

  // reset to zeros
  std::memset(dst, 0, _size);

#pragma omp parallel for simd
  for (size_t i = 0; i < this->_k; ++i) {
    dst[_selected_idx[i]] = buf[i];
  }

  _selected_idx.clear();
  return {dst, _size};
#else
  // do nothing
  return {dst, compressed_size};
#endif
}  // namespace compressor

tensor_t RandomkCompressor::Decompress(tensor_t compressed) {
  auto dst = compressed.data;
  DECOMPRESS_IMPL_SWITCH2(_dtype, DecompressImpl, dst, compressed.data,
                          compressed.size);
}

template <typename scalar_t>
void RandomkCompressor::FastUpdateErrorImpl(scalar_t* error,
                                            scalar_t* corrected,
                                            const scalar_t* compressed,
                                            size_t compressed_size) {
  memcpy_multithread(error, corrected, _size);

#pragma omp parallel for simd
  for (size_t i = 0; i < this->_k; ++i) {
    error[_selected_idx[i]] = 0;
  }
}

void RandomkCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                        tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH2(_dtype, FastUpdateErrorImpl, error.data,
                                 corrected.data, compressed.data,
                                 compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps