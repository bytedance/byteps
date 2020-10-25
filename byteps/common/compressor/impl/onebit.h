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

#ifndef BYTEPS_COMPRESSOR_IMPL_ONEBIT_H
#define BYTEPS_COMPRESSOR_IMPL_ONEBIT_H

#include "../compressor.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief Onebit Compressor
 *
 * paper: SIGNSGD: Compressed Optimisation for Non-Convex Problems
 * https://arxiv.org/pdf/1802.04434.pdf
 *
 * each worker i:
 *    c_i <- sign(grad)
 *
 * server: majority vote
 *    sign(\sum_i c_i)
 *
 * \note 0 represents positive and 1 represents negative.
 */
class OnebitCompressor : public Compressor {
 public:
  OnebitCompressor(size_t size, DataType dtype, bool use_scale = false)
      : Compressor(size, dtype), _use_scale(use_scale) {}
  ~OnebitCompressor() override = default;

  /*!
   * \brief Compress function
   *
   * compress and pack into byte array.
   * each bit represents a sign.
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  void Compress(tensor_t grad, tensor_t& output) override;

  /*!
   * \brief Decompress function
   *
   * unpack from byte array to FP tensor
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  void Decompress(tensor_t compressed, tensor_t& output) override;

  /*!
   * \brief help function for error feedback `UpdateError`
   *
   * \param corrected gradient corrected with error
   * \param error error
   * \param compressed compressed gradient
   */
  void FusedCompress(tensor_t grad, tensor_t& output, tensor_t error) override;

 private:
  template <typename index_t, typename scalar_t>
  size_t CompressImpl(index_t* __restrict__ dst,
                      const scalar_t* __restrict__ src, size_t len);

  template <typename scalar_t, typename index_t>
  void DecompressImpl(scalar_t* __restrict__ dst,
                      const index_t* __restrict__ src, size_t compressed_size,
                      size_t dst_size);

  template <typename index_t, typename scalar_t>
  size_t FusedCompressImpl(index_t* __restrict__ dst,
                           const scalar_t* __restrict__ src,
                           scalar_t* __restrict__ error, size_t len);

 private:
  bool _use_scale;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_ONEBIT_H