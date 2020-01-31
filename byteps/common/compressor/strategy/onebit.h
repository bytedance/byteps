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

#ifndef BYTEPS_COMPRESS_STRAT_ONEBIT_H
#define BYTEPS_COMPRESS_STRAT_ONEBIT_H

#include "../base_compressor.h"

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
 * server:
 *    sign(\sum_i c_i) 
 * 
 * \note this is a deterministic algorithm. 
 */
class OnebitCompressor : public BaseCompressor {
 public:
  OnebitCompressor();
  virtual ~OnebitCompressor();
  
  /*!
   * \brief Compress
   * 
   * compress and pack into byte array.
   * each bit represents a sign.
   * 
   * \param grad 
   * \return ByteBuf: byte array
   */
  ByteBuf Compress(const ByteBuf& grad) override;

  /*!
   * \brief Decompress
   * 
   * unpack from byte array to FP tensor
   * 
   * \param compressed 
   * \return ByteBuf: tensor
   */
  ByteBuf Decompress(const ByteBuf& compressed) override;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESS_STRAT_ONEBIT_H