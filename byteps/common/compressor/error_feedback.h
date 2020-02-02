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
   * \brief Allocate all buffers for compression.
   * \param size the size of buffer (bytes)
   */
  void AllocateBuffer(size_t size) final;

  /*!
   * \brief Compress with error feedback
   *
   *  invoke `UpdateGradient` and `UpdateError` before and after
   *  `_compressor_ptr->Compress(*)`
   */
  ByteBuf Compress(const ByteBuf& grad) final;

  /*!
   * \brief Decompress
   *
   *  directly forward to `_compressor_ptr->Decompress(*)`
   */
  ByteBuf Decompress(const ByteBuf& compressed) final;

 protected:
  /*!
   * \brief Correct gradient with error
   *
   * grad += error
   *
   * \param grad input gradient to be updated
   * \return corrected gradient
   */
  virtual ByteBuf UpdateGradient(const ByteBuf& grad) = 0;

  /*!
   * \brief Update error
   *
   * error = corrected_grad - decompress(compressed_corrected_grad)
   *
   * \param corrected refers to gradient + error
   */
  virtual void UpdateError(const ByteBuf& corrected,
                           const ByteBuf& compressed) = 0;

 protected:
  std::unique_ptr<char[]> _decode_buf;
  std::unique_ptr<char[]> _error_buf;

 private:
  /*!
   * \brief compressor
   */
  std::unique_ptr<BaseCompressor> _compressor_ptr;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESS_ERROR_FEEDBACK_H