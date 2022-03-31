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

#include "efsignsgd.h"
#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg("efsignsgd_compressor", [](const kwargs_t& kwargs,
                                                         size_t size,
                                                         DataType dtype) {
  auto scaled =
      HyperParamFinder<bool>(kwargs, "compressor_efsignsgd_scaling", true);
  return std::unique_ptr<Compressor>(new EFSignSGDCompressor(Align(size, dtype), dtype, size/sizeof(float), scaled));
});

template <typename type_t> 
double getMean(const type_t *array, size_t len) {
  double sum = 0.0f; 
  #pragma omp parallel for simd reduction(+: sum)
  for (size_t i=0; i<len; i++) {
      sum += std::abs(array[i]);
  }
  return sum / len;
}

}


template <typename index_t, typename scalar_t>
tensor_t EFSignSGDCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                        size_t numel) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");

  auto dst_ptr = reinterpret_cast<uint32_t*>(dst);
  constexpr size_t PACKING_SIZE = sizeof(uint32_t) * 8;
  size_t padding_len = (PACKING_SIZE - (_numel % PACKING_SIZE)) % PACKING_SIZE;
  const size_t chunk_len = (_numel + padding_len) / PACKING_SIZE;
  
  float scale = 1.0f;
  if (_use_scale) {
    scale = getMean<scalar_t>(src, _numel);
  }

#pragma omp parallel for simd num_threads(_num_threads)
  for (size_t i = 0; i < chunk_len; ++i) {
    index_t x = src[i * PACKING_SIZE] > 0;
    for (size_t j = 1; j < PACKING_SIZE; ++j) {
      x <<= 1;
      x |= src[i * PACKING_SIZE + j] > 0;
    }
    dst_ptr[i] = x;
  }

  float* p_scale = reinterpret_cast<float*>(&dst[chunk_len]);
  *p_scale = scale;

  BPS_LOG(INFO) << "EFSignSGDCompressor CompressImpl numel: " << _numel
  << ", len: " << numel
  << ", chunk len: " << chunk_len
  << ", scale: " << scale
  << std::endl;

  return {dst_ptr, chunk_len * sizeof(uint32_t) + sizeof(float)};
}


tensor_t EFSignSGDCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}


template <typename scalar_t, typename index_t>
tensor_t EFSignSGDCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                          size_t compressed_size) {
  static_assert(sizeof(scalar_t) == sizeof(index_t),
                "scalar_t should be the same size as index_t");
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  const size_t chunk_len = (compressed_size - sizeof(float)) / sizeof(index_t);

  const float* float_src = reinterpret_cast<const float*>(src);
  float scale = float_src[chunk_len];
  index_t* ptr = const_cast<index_t*>(src);

#pragma omp parallel for simd num_threads(_num_threads)
  for (int i = chunk_len - 1; i >= 0; --i) {
    index_t x = ptr[i];
    for (int j = PACKING_SIZE - 1; j >= 0; --j) {
      dst[i * PACKING_SIZE + j] = (x & 0x01) ? scale : -scale;
      x >>= 1;
    }
  }

  BPS_LOG(INFO) << "EFSignSGDCompressor DecompressImpl numel: " << _numel
  << ", compressed size: " << compressed_size
  << ", chunk len: " << chunk_len
  << ", scale: " << scale
  << std::endl;

  return {dst, _numel * sizeof(float)};
}

tensor_t EFSignSGDCompressor::Decompress(tensor_t compressed) {
  DECOMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, DecompressImpl, _buf.get(), compressed.data,
                         compressed.size);
}


template <typename scalar_t, typename index_t>
void EFSignSGDCompressor::FastUpdateErrorImpl(scalar_t* error, scalar_t* corrected,
                                           const index_t* compressed,
                                           size_t compressed_size) {  
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  const size_t chunk_len = (compressed_size - sizeof(float)) / sizeof(index_t);

  auto* pf = reinterpret_cast<const float*>(compressed + chunk_len);
  float scale = *pf;

#pragma omp parallel for simd num_threads(_num_threads)
  for (int i = chunk_len - 1; i >= 0; --i) {
    index_t x = compressed[i];
    for (int j = PACKING_SIZE - 1; j >= 0; --j) {
      int sign = ((x & 0x01) << 1) - 1;
      error[i * PACKING_SIZE + j] =
          corrected[i * PACKING_SIZE + j] - sign * scale;
      x >>= 1;
    }
  }
}

void EFSignSGDCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                       tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(BYTEPS_FLOAT32, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps
