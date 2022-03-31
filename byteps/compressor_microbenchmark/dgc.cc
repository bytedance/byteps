#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <algorithm> 
#include "dgc.h"

namespace byteps {
namespace common {
namespace compressor {

template <typename index_t, typename scalar_t>
tensor_t DGCCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                         size_t len) {
  // no matter src is FP16 or FP32, the value is FP32 after compression
  using pair_t = std::pair<int, float>;
  auto ptr = reinterpret_cast<pair_t*>(dst);
  float _sample_ratio = 0.001;
  if (this->_numel < 5*1024*1024) {
    _sample_ratio = 0.01;
  } else if (this->_numel < 1024*10) {
    _sample_ratio = 0.1;
  } else if (this->_numel < 1024) {
    _sample_ratio = 1;
  }

  size_t sample_num = (int) (this->_numel * _sample_ratio);
  int sample_step = (int) (1 / _sample_ratio);
  int offset = rand() % sample_step;

  float* sample_set = reinterpret_cast<float*>(dst);

  //auto start = std::chrono::high_resolution_clock::now();
// step 1: sample the input
#pragma omp parallel for simd num_threads(_num_threads)
  for (int i=0; i<sample_num; i++) {
    float value = src[i*sample_step + offset];
    sample_set[i] = std::abs(value);
  }
  //auto step1 = std::chrono::high_resolution_clock::now();

// step 2: get the threshold
  size_t sample_k = std::max(1, (int) (sample_num * this->_compress_ratio * 2));
  std::nth_element(sample_set, sample_set+sample_k, sample_set+sample_num, std::greater<float>());
  auto threshold = sample_set[sample_k]; 
  if (!std::isfinite(threshold)) {
    threshold = 0;
  }

// step 3: get the elements larger than threshold
  size_t total_k = std::ceil(_numel * _compress_ratio);
  size_t threads = 1;
  size_t thread_k = total_k / threads;
  memset(dst, 0, total_k * sizeof(pair_t));
  //auto step2 = std::chrono::high_resolution_clock::now();

  int thread_cnt = 0;
  pair_t* thread_ptr = nullptr;

#pragma omp parallel private(thread_ptr, thread_cnt) num_threads(threads)
{
  thread_cnt = 0;
  thread_ptr = ptr + thread_k * omp_get_thread_num();
  #pragma omp for
  for (uint32_t i=0; i<this->_numel; i++) {
    float value = src[i];
    if (thread_cnt < thread_k && std::isfinite(value) && std::abs(value) >= threshold) {
      thread_ptr[thread_cnt++] = std::make_pair(i, value);
    }
  }
}

  return {dst, total_k * sizeof(pair_t)};
}


tensor_t DGCCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename index_t, typename scalar_t>
tensor_t DGCCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                           size_t compressed_size) {
  using pair_t = std::pair<int, float>;

  auto ptr = reinterpret_cast<const pair_t*>(src);
  // reset to zeros
  // TODO: optimize the summation of sparsification
  std::memset(dst, 0, _numel*sizeof(float));
  size_t len = compressed_size / sizeof(pair_t);
   
#pragma omp parallel for simd
  for (size_t i = 0; i < len; ++i) {
    auto& pair = ptr[i];
    dst[pair.first] = pair.second;
  }

  return {dst, _numel * sizeof(float)};
}

tensor_t DGCCompressor::Decompress(tensor_t compressed) {
  DECOMPRESS_IMPL_SWITCH(BYTEPS_FLOAT32, DecompressImpl, _buf.get(), compressed.data,
                         compressed.size);
}


template <typename index_t, typename scalar_t>
void DGCCompressor::FastUpdateErrorImpl(scalar_t* error,
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

void DGCCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                        tensor_t compressed) {
  FAST_UPDATE_ERROR_IMPL_SWITCH(BYTEPS_FLOAT32, FastUpdateErrorImpl, error.data,
                                corrected.data, compressed.data,
                                compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps