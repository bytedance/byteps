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

#ifndef BYTEPS_COMPRESS_STRAT_RANDOMK_H
#define BYTEPS_COMPRESS_STRAT_RANDOMK_H

#include <random>

#include "../base_compressor.h"

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
 */
class RandomkCompressor : public BaseCompressor {
 public:
  explicit RandomkCompressor(int k);
  virtual ~RandomkCompressor();

  /*!
   * \brief Compress function
   *
   * randomly select k entries and corresponding indices
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  void Compress(tensor_t grad, tensor_t& compressed) override;

  /*!
   * \brief Decompress function
   *
   * fill a zero tensor with topk entries and corresponding indices
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  void Decompress(tensor_t compressed, tensor_t& decompressed) override;

  /*!
   * \brief help function for error feedback `UpdateError`
   *
   * \param corrected gradient corrected with error
   * \param error error
   * \param compressed compressed gradient
   */
  void FastUpdateError(tensor_t error, tensor_t corrected,
                       tensor_t compressed) override;

 private:
  size_t Packing(const void* src, size_t size, int dtype);

  template <typename index_t, typename scalar_t>
  size_t PackingImpl(index_t* dst, const scalar_t* src, size_t len);

  void Unpacking(void* dst, const void* src, size_t size, size_t src_size,
                 int dtype);

  template <typename index_t, typename scalar_t>
  void UnpackingImpl(scalar_t* dst, const index_t* src, size_t len,
                     size_t src_len);

  template <typename index_t, typename scalar_t>
  void FastUpdateErrorImpl(scalar_t* error, const index_t* compressed,
                           size_t len);

 private:
  int _k;
  std::random_device _rd;
  std::mt19937 _gen;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESS_STRAT_RANDOMK_H