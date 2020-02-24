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

#ifndef BYTEPS_COMPRESS_ERROR_FEEDBACK_H
#define BYTEPS_COMPRESS_ERROR_FEEDBACK_H

#include "base_compressor.h"

#include <future>

namespace byteps {
namespace common {
namespace compressor {

/*!
 *  \brief Error feedback Decorator
 *
 *  add error feedback behavior to any compressor at run-time
 */
class ErrorFeedback : public BaseCompressor {
 public:
  explicit ErrorFeedback(std::unique_ptr<BaseCompressor> compressor_ptr);
  virtual ~ErrorFeedback();

  /*!
   * \brief Allocate encoding buffer for compression.
   * \param aligned_size aligned size
   */
  virtual void Init(size_t aligned_size) final;

  /*!
   * \brief Compress function
   *
   * \param grad gradient tensor
   * \param dtype data type
   * \param compressed compressed tensor
   */
  virtual void Compress(ByteBuf grad, int dtype, ByteBuf* compressed) final;

  /*!
   * \brief Decompress function
   *
   * \param compressed compressed tensor
   * \param dtype data type
   * \param decompressed decompressed tensor
   */
  virtual void Decompress(ByteBuf compressed, int dtype,
                          ByteBuf* decompressed) final;

 protected:
  /*!
   * \brief Correct gradient with error
   *
   * grad += error
   *
   * \param grad input gradient to be updated inplace
   * \param dtype type
   */
  virtual void UpdateGradient(ByteBuf grad, int dtype) = 0;

  /*!
   * \brief Update error
   *
   * error = corrected_grad - decompressed
   *
   * \param corrected refers to gradient + error
   * \param dtype type
   * \param compressed compressed tensor
   */
  virtual void UpdateError(ByteBuf corrected, int dtype,
                           ByteBuf* decompressed) = 0;

 protected:
  std::unique_ptr<char[]> _decode_buf;
  std::unique_ptr<char[]> _error;

 private:
  /*!
   * \brief compressor
   */
  std::unique_ptr<BaseCompressor> _compressor_ptr;

  std::future<void> _future;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESS_ERROR_FEEDBACK_H