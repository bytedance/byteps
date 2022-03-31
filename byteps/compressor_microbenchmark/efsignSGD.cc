#include <iostream>
#include <cstring>
#include <omp.h>
#include "efsignSGD.h"

namespace byteps {
namespace common {
namespace compressor {

namespace {
    template <typename type_t> 
    double getMean(const type_t *array, size_t len) {
      double sum = 0.0f; 
      #pragma omp parallel for reduction(+: sum) num_threads(8)
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

#pragma omp parallel for simd
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