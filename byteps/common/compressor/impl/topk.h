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

#ifndef BYTEPS_COMPRESSOR_IMPL_TOPK_H
#define BYTEPS_COMPRESSOR_IMPL_TOPK_H

#include "../compressor.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief TopK Compressor
 *
 * paper: Sparsified SGD with Memory
 * https://arxiv.org/pdf/1809.07599.pdf
 *
 * sending the most significant entries of the stochastic gradient
 *
 */
class TopkCompressor : public Compressor {
 public:
  TopkCompressor(size_t size, DataType dtype, unsigned int k)
      : Compressor(size, dtype), _k(k){};
  ~TopkCompressor() override = default;

  void Compress(tensor_t grad, tensor_t& output) override;

  void Decompress(tensor_t compressed, tensor_t& output) override;

  void FusedCompress(tensor_t grad, tensor_t& output, tensor_t error) override;

 private:
  template <typename pair_t, typename scalar_t>
  size_t CompressImpl(pair_t* __restrict__ dst,
                      const scalar_t* __restrict__ src, size_t len);

  template <typename scalar_t, typename pair_t>
  void DecompressImpl(scalar_t* __restrict__ dst,
                      const pair_t* __restrict__ src, size_t compressed_size,
                      size_t dst_size);

  template <typename pair_t, typename scalar_t>
  size_t FusedCompressImpl(pair_t* __restrict__ dst,
                           const scalar_t* __restrict__ src,
                           scalar_t* __restrict__ error, size_t len);

 private:
  unsigned int _k;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_TOPK_H