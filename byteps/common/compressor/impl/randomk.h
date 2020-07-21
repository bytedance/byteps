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
  RandomkCompressor(size_t size, DataType dtype, unsigned int k, unsigned int seed = 0)
      : Compressor(size, dtype), _k(k) {
    if (seed != 0) {
      BPS_LOG(INFO) << "SET SEED = " << seed;
      _rng.set_seed(seed);
    }
  };
  virtual ~RandomkCompressor() = default;

  /*!
   * \brief Compress function
   *
   * randomly select k entries and corresponding indices
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  tensor_t Compress(tensor_t grad) override;

  /*!
   * \brief Decompress function
   *
   * fill a zero tensor with topk entries and corresponding indices
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  tensor_t Decompress(tensor_t compressed) override;

  /*!
   * \brief faster version of `UpdateError`
   *
   * 1. e <- p (e is the error and p is the corrected gradient)
   * 2. zero-fill e with selected k indices
   *
   * \param corrected gradient corrected with error
   * \param error error
   * \param compressed compressed gradient
   */
  void FastUpdateError(tensor_t error, tensor_t corrected,
                       tensor_t compressed) override;

 private:
  template <typename index_t, typename scalar_t>
  tensor_t CompressImpl(index_t* dst, const scalar_t* src, size_t len);

  template <typename index_t, typename scalar_t>
  tensor_t DecompressImpl(scalar_t* dst, const index_t* src,
                          size_t compressed_size);

  template <typename index_t, typename scalar_t>
  void FastUpdateErrorImpl(scalar_t* error, scalar_t* corrected,
                           const index_t* compressed, size_t compressed_size);

 private:
  unsigned int _k;
  std::random_device _rd;
  XorShift128PlusBitShifterRNG _rng;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_RANDOMK_H