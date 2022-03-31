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

#include "onebit.h"
#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg("onebit_compressor", [](const kwargs_t& kwargs,
                                                         size_t size,
                                                         DataType dtype) {
  auto scaled =
      HyperParamFinder<bool>(kwargs, "compressor_onebit_scaling", true);
  return std::unique_ptr<Compressor>(new OnebitCompressor(Align(size, dtype), dtype, size/sizeof(float), scaled));
});

template <typename type_t> 
std::vector<double> getMean(const type_t *array, size_t len) {
  // sum1: the sum of items >= 0; sum2: the sum of items < 0
  double sum1 = 0.0f, sum2 = 0.0f;
  // cnt1: the number of items >= 0; cnt2: the number of items < 0
  int cnt1 = 0, cnt2 = 0; 
  #pragma omp parallel for simd reduction(+: sum1, sum2, cnt1, cnt2)
  for (size_t i=0; i<len; i++) {
    if (array[i] >= 0) {
      sum1 += array[i];
      cnt1 += 1;
    } else {
      sum2 += array[i];
      cnt2 += 1;
    }
  }
  return {sum1 / cnt1, sum2 / cnt2};
}

}


template <typename index_t, typename scalar_t>
tensor_t OnebitCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                        size_t numel) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t"); 
  auto dst_ptr = reinterpret_cast<uint32_t*>(dst);
  constexpr size_t PACKING_SIZE = sizeof(uint32_t) * 8;
  size_t padding_len = (PACKING_SIZE - (_numel % PACKING_SIZE)) % PACKING_SIZE;
  const size_t chunk_len = (_numel + padding_len) / PACKING_SIZE;
  
  float scale1 = 1.0f, scale2 = -1.0f;
  if (_use_scale) {
    auto scales = getMean<scalar_t>(src, _numel);
    scale1 = scales[0];
    scale2 = scales[1];
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
  *p_scale = scale1;
  *(p_scale + 1) = scale2;

  BPS_LOG(DEBUG) << "OneBitCompressor CompressImpl numel: " << _numel
  << ", len: " << numel
  << ", chunk len: " << chunk_len
  << ", scale: " << scale1 << " " << scale2
  << std::endl;

  return {dst_ptr, chunk_len * sizeof(uint32_t) + 2 * sizeof(float)};
}  // namespace compressor

tensor_t OnebitCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename scalar_t, typename index_t>
tensor_t OnebitCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                          size_t compressed_size) {
  static_assert(sizeof(scalar_t) == sizeof(index_t),
                "scalar_t should be the same size as index_t");
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  const size_t chunk_len = (compressed_size - 2 * sizeof(float)) / sizeof(index_t);
  
  const float* float_src = reinterpret_cast<const float*>(src);
  float scale1 = float_src[chunk_len];
  float scale2 = float_src[chunk_len+1];
  index_t* ptr = const_cast<index_t*>(src);

#pragma omp parallel for simd num_threads(_num_threads)
  for (int i = chunk_len - 1; i >= 0; --i) {
    index_t x = ptr[i];
    for (int j = PACKING_SIZE - 1; j >= 0; --j) {
      dst[i * PACKING_SIZE + j] = (x & 0x01) ? scale1 : scale2;
      x >>= 1;
    }
  }

  BPS_LOG(DEBUG) << "OneBitCompressor DecompressImpl numel: " << _numel
  << ", compressed size: " << compressed_size
  << ", chunk len: " << chunk_len
  << ", scale: " << scale1 << " " << scale2
  << std::endl;

  return {dst, _numel * sizeof(float)};
}

tensor_t OnebitCompressor::Decompress(tensor_t compressed) {
  DECOMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, DecompressImpl, _buf.get(), compressed.data,
                         compressed.size);
}


template <typename scalar_t, typename index_t>
void OnebitCompressor::FastUpdateErrorImpl(scalar_t* error, scalar_t* corrected,
                                           const index_t* compressed,
                                           size_t compressed_size) {
  constexpr size_t PACKING_SIZE = sizeof(index_t) * 8;
  const size_t chunk_len = (compressed_size - 2 * sizeof(float)) / sizeof(index_t);

  auto* pf = reinterpret_cast<const float*>(compressed + chunk_len);
  float scale1 = *pf;
  float scale2 = *(pf + 1);

#pragma omp parallel for simd num_threads(_num_threads)
  for (int i = chunk_len - 1; i >= 0; --i) {
    index_t x = compressed[i];
    for (int j = PACKING_SIZE - 1; j >= 0; --j) {
      float value = (x & 0x01) ? scale1 : scale2;
      error[i * PACKING_SIZE + j] =
          corrected[i * PACKING_SIZE + j] - value;
      x >>= 1;
    }
  }
}

void OnebitCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                       tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(BYTEPS_FLOAT32, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps