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

#ifndef BYTEPS_COMPRESSOR_IMPL_MULTIBIT_H
#define BYTEPS_COMPRESSOR_IMPL_MULTIBIT_H

#include "../compressor.h"
#include "../utils.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief Dithering Compressor
 *
 * paper: Natural Compression for Distributed Deep Learning
 * https://arxiv.org/pdf/1905.10988.pdf
 *
 * two kinds of partition:
 * 1. linear: {0, 1/s, 2/s, ..., (s-1)/s, 1}
 *
 * 2. natural: {0, 2^{1-s}, 2^(2-s), ..., 2^{-1}, 1}
 *
 * two kinds of normalization:
 * 1. max: it gives better accuracy but less sparsity.
 *
 * 2. l2 norm: it is more sparse but less accurate. and
 * empirically we found it will diverge with error-feedback.
 */
class DitheringCompressor : public Compressor {
 public:
  enum class PartitionType { LINEAR = 0, NATURAL = 1 };
  enum class NormalizeType { MAX = 0, L2 = 1 };

  DitheringCompressor(size_t size, DataType dtype, unsigned int s,
                      unsigned int seed = 0,
                      PartitionType ptype = PartitionType::LINEAR,
                      NormalizeType ntype = NormalizeType::MAX)
      : Compressor(size, dtype), _s(s), _ptype(ptype), _ntype(ntype) {
    if (seed) {
      _rng.set_seed(seed);
    }
  };
  ~DitheringCompressor() override;

  tensor_t Compress(tensor_t grad) override;

  tensor_t Decompress(tensor_t compressed) override;

  void FastUpdateError(tensor_t error, tensor_t corrected,
                       tensor_t compressed) override;

 private:
  template <typename index_t, typename scalar_t>
  tensor_t CompressImpl(index_t* __restrict__ dst,
                        const scalar_t* __restrict__ src, size_t len);

  template <typename index_t, typename scalar_t>
  tensor_t CompressImplMax(index_t* __restrict__ dst,
                           const scalar_t* __restrict__ src, size_t len);

  template <typename index_t, typename scalar_t>
  tensor_t CompressImplL2(index_t* __restrict__ dst,
                          const scalar_t* __restrict__ src, size_t len);

  template <typename index_t, typename scalar_t>
  tensor_t DecompressImplL2(scalar_t* __restrict__ dst,
                            const index_t* __restrict__ src,
                            size_t compressed_size);

  template <typename index_t, typename scalar_t>
  tensor_t DecompressImplMax(scalar_t* __restrict__ dst,
                             const index_t* __restrict__ src,
                             size_t compressed_size);

  template <typename index_t, typename scalar_t>
  tensor_t DecompressImpl(scalar_t* __restrict__ dst,
                          const index_t* __restrict__ src,
                          size_t compressed_size);

  template <typename index_t, typename scalar_t>
  void FastUpdateErrorImplL2(scalar_t* __restrict__ error,
                             scalar_t* __restrict__ corrected,
                             const index_t* __restrict__ compressed,
                             size_t compressed_size);

  template <typename index_t, typename scalar_t>
  void FastUpdateErrorImplMax(scalar_t* __restrict__ error,
                              scalar_t* __restrict__ corrected,
                              const index_t* __restrict__ compressed,
                              size_t compressed_size);

  template <typename index_t, typename scalar_t>
  void FastUpdateErrorImpl(scalar_t* __restrict__ error,
                           scalar_t* __restrict__ corrected,
                           const index_t* __restrict__ compressed,
                           size_t compressed_size);

  /*! \brief number of levels */
  const unsigned int _s;

  PartitionType _ptype;
  NormalizeType _ntype;
  XorShift128PlusBitShifterRNG _rng;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_MULTIBIT_H