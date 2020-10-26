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
size_t RandomkCompressor::CompressImpl(scalar_t* __restrict__ dst,
                                       const scalar_t* __restrict__ src,
                                       size_t len) {
  for (size_t i = 0; i < this->_k; ++i) {
    _selected_idx.push_back(_rng.Randint(0, len));
  }

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
  // server does nothing
  return this->_k * sizeof(scalar_t);
}  // namespace compressor

void RandomkCompressor::Compress(tensor_t grad, tensor_t& output) {
#ifndef BYTEPS_BUILDING_SERVER
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.data != output.data);
  if (output.data == nullptr) {
    output.data = _buf.get();
  }

  size_t compressed_size;
  switch (grad.dtype) {
    case BYTEPS_FLOAT16:
      compressed_size = CompressImpl(reinterpret_cast<half_t*>(output.data),
                                     reinterpret_cast<const half_t*>(grad.data),
                                     grad.size / sizeof(half_t));
      break;
    case BYTEPS_FLOAT32:
      compressed_size = CompressImpl(reinterpret_cast<float*>(output.data),
                                     reinterpret_cast<const float*>(grad.data),
                                     grad.size / sizeof(float));
      break;
    case BYTEPS_FLOAT64:
      compressed_size = CompressImpl(reinterpret_cast<double*>(output.data),
                                     reinterpret_cast<const double*>(grad.data),
                                     grad.size / sizeof(double));
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
  }

  output.size = compressed_size;
#else
  output = grad;
#endif
}

template <typename scalar_t>
void RandomkCompressor::DecompressImpl(scalar_t* __restrict__ dst,
                                       const scalar_t* __restrict__ src,
                                       size_t compressed_size,
                                       size_t dst_size) {
  // reset to zeros
  std::memset(dst, 0, dst_size);

#pragma omp parallel for simd
  for (size_t i = 0; i < this->_k; ++i) {
    dst[_selected_idx[i]] = src[i];
  }

  _selected_idx.clear();
}

void RandomkCompressor::Decompress(tensor_t compressed, tensor_t& output) {
#ifndef BYTEPS_BUILDING_SERVER
  BPS_CHECK(compressed.data);
  BPS_CHECK(compressed.data != output.data);
  BPS_CHECK(output.data);

  switch (output.dtype) {
    case BYTEPS_FLOAT16:
      DecompressImpl(reinterpret_cast<half_t*>(output.data),
                     reinterpret_cast<const half_t*>(compressed.data),
                     compressed.size, output.size);
      break;
    case BYTEPS_FLOAT32:
      DecompressImpl(reinterpret_cast<float*>(output.data),
                     reinterpret_cast<const float*>(compressed.data),
                     compressed.size, output.size);
      break;
    case BYTEPS_FLOAT64:
      DecompressImpl(reinterpret_cast<double*>(output.data),
                     reinterpret_cast<const double*>(compressed.data),
                     compressed.size, output.size);
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << output.dtype;
  }
#else
  output = compressed;
#endif
}

template <typename scalar_t>
size_t RandomkCompressor::FusedCompressImpl(scalar_t* __restrict__ dst,
                                            const scalar_t* __restrict__ src,
                                            scalar_t* __restrict__ error,
                                            size_t len) {
  for (size_t i = 0; i < this->_k; ++i) {
    _selected_idx.push_back(_rng.Randint(0, len));
  }

  memcpy_multithread(error, src, len * sizeof(scalar_t));

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

#pragma omp parallel for simd
  for (size_t i = 0; i < this->_k; ++i) {
    error[_selected_idx[i]] = 0;
  }

  return this->_k * sizeof(scalar_t);
}

void RandomkCompressor::FusedCompress(tensor_t grad, tensor_t& output,
                                      tensor_t error) {
#ifndef BYTEPS_BUILDING_SERVER
  BPS_CHECK(grad.data);
  BPS_CHECK(grad.data != output.data);
  if (output.data == nullptr) {
    output.data = _buf.get();
  }

  size_t compressed_size;
  switch (grad.dtype) {
    case BYTEPS_FLOAT16:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<half_t*>(output.data),
          reinterpret_cast<const half_t*>(grad.data),
          reinterpret_cast<half_t*>(error.data), grad.size / sizeof(half_t));
      break;
    case BYTEPS_FLOAT32:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<float*>(output.data),
          reinterpret_cast<const float*>(grad.data),
          reinterpret_cast<float*>(error.data), grad.size / sizeof(float));
      break;
    case BYTEPS_FLOAT64:
      compressed_size = FusedCompressImpl(
          reinterpret_cast<double*>(output.data),
          reinterpret_cast<const double*>(grad.data),
          reinterpret_cast<double*>(error.data), grad.size / sizeof(double));
      break;
    default:
      BPS_CHECK(0) << "Unsupported data type:" << grad.dtype;
  }

  output.size = compressed_size;
#else
  output = grad;
#endif
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps