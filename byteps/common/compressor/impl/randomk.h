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

#ifndef BYTEPS_COMPRESSOR_IMPL_RANDOMK_H
#define BYTEPS_COMPRESSOR_IMPL_RANDOMK_H

#include <random>
#include <vector>

#include "../compressor.h"
#include "../utils.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief RandomK Compressor
 *
 * paper: Sparsified SGD with Memory
 * https://arxiv.org/pdf/1809.07599.pdf
 *
 * randomly sending k entries of the stochastic gradient
 *
 * \note it is a stochastic algorithm. If you want to have deterministic
 * behavior, please set a seed in the configurations.
 */
class RandomkCompressor : public Compressor {
 public:
  RandomkCompressor(size_t size, DataType dtype, unsigned int k,
                    unsigned int seed = 0, bool is_scale = false)
      : Compressor(size, dtype), _k(k), _is_scale(is_scale) {
    if (seed) {
      BPS_LOG(INFO) << "SET SEED = " << seed + k;
      _rng.set_seed(seed + k);
    }
  };
  ~RandomkCompressor() override = default;

  void Compress(tensor_t grad, tensor_t& output) override;

  void Decompress(tensor_t compressed, tensor_t& output) override;

  void FusedCompress(tensor_t grad, tensor_t& output, tensor_t error) override;

 private:
  template <typename scalar_t>
  size_t CompressImpl(scalar_t* __restrict__ dst,
                      const scalar_t* __restrict__ src, size_t len);

  template <typename scalar_t>
  void DecompressImpl(scalar_t* __restrict__ dst,
                      const scalar_t* __restrict__ src, size_t compressed_size,
                      size_t dst_size);

  template <typename scalar_t>
  size_t FusedCompressImpl(scalar_t* __restrict__ dst,
                           const scalar_t* __restrict__ src,
                           scalar_t* __restrict__ error, size_t len);

 private:
  unsigned int _k;

  XorShift128PlusBitShifterRNG _rng;
  std::vector<uint32_t> _selected_idx;
  bool _is_scale;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_RANDOMK_H