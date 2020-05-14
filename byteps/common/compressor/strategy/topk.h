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

#ifndef BYTEPS_COMPRESS_STRAT_TOPK_H
#define BYTEPS_COMPRESS_STRAT_TOPK_H

#include "../base_compressor.h"

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
 * \note this is a deterministic algorithm
 */
class TopkCompressor : public BaseCompressor {
 public:
  explicit TopkCompressor(int k);
  virtual ~TopkCompressor();
  
  /*!
   * \brief Compress function
   * 
   * select topk entries and corresponding indices 
   * 
   * \note compare with absolute values
   * 
   * \param grad gradient tensor
   * \param dtype data type
   * \param compressed compressed tensor
   */
  void Compress(ByteBuf grad, int dtype, ByteBuf& compressed) override;

  /*!
   * \brief Decompress function
   * 
   * fill a zero tensor with topk entries and corresponding indices 
   * 
   * \param compressed compressed tensor
   * \param dtype data type
   * \param decompressed decompressed tensor
   */
  void Decompress(ByteBuf compressed, int dtype,
                  ByteBuf& decompressed) override;

 private:
  size_t Packing(const void* src, size_t size, int dtype);

  template <typename index_t, typename scalar_t>
  size_t _Packing(index_t* dst, const scalar_t* src, size_t len);

  size_t Unpacking(void* dst, const void* src, size_t size, int dtype);
  
  template <typename index_t, typename scalar_t>
  size_t _Unpacking(scalar_t* dst, const index_t* src, size_t len);

 private:
  int _k;
  int _src_len;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESS_STRAT_MULTIBIT_H