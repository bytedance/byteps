#include <iostream>
#include <cstring>
#include <omp.h>
#include <cmath>
#include "randomk.h"

namespace byteps {
namespace common {
namespace compressor {

template <typename index_t, typename scalar_t>
tensor_t RandomkCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                         size_t len) {

  using pair_t = std::pair<int, float>;
  auto ptr = reinterpret_cast<pair_t*>(dst);
  size_t k = std::ceil(_numel * _compress_ratio);

#pragma omp parallel for simd
  for (size_t i = 0; i < k; ++i) {
    auto index = _rng.Randint(0, _numel);
    ptr[i] = std::make_pair(index, src[index]);
  }

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

#pragma omp parallel for simd
  for (size_t i = 0; i < numel; ++i) {
    auto& pair = ptr[i];
    dst[pair.first] = pair.second;
  }

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
#pragma omp parallel for simd
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