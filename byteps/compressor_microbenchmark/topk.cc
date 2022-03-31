#include <iostream>
#include <cstring>
#include <omp.h>
#include "topk.h"

namespace byteps {
namespace common {
namespace compressor {

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

#pragma omp parallel for simd
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