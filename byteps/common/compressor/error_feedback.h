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

#ifndef BYTEPS_COMPRESSOR_ERROR_FEEDBACK_H
#define BYTEPS_COMPRESSOR_ERROR_FEEDBACK_H

#include "compressor.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 *  \brief Error feedback Decorator
 *
 *  add error feedback behavior to any compressor at run-time
 */
class ErrorFeedback : public Compressor {
 public:
  ErrorFeedback(size_t size, std::unique_ptr<Compressor> cptr)
      : Compressor(size), _cptr(std::move(cptr)), _error(new byte_t[size]()) {}
  virtual ~ErrorFeedback() = default;

  /*!
   * \brief Compress function
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  virtual void Compress(tensor_t grad, tensor_t& compressed);

  /*!
   * \brief Decompress function
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  virtual void Decompress(tensor_t compressed, tensor_t& decompressed);

 protected:
  /*!
   * \brief Correct gradient with error
   *
   * grad += error
   *
   * \param grad input gradient to be updated inplace
   * \param dtype type
   */
  virtual void UpdateGradient(tensor_t grad) = 0;

  /*!
   * \brief Update error
   *
   * error = corrected_grad - decompressed
   *
   * \param corrected refers to gradient + error
   * \param compressed compressed tensor
   */
  virtual void UpdateError(tensor_t corrected, tensor_t compressed);

 protected:
  std::unique_ptr<byte_t[]> _error;

 private:
  /*!
   * \brief compressor
   */
  std::unique_ptr<Compressor> _cptr;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_ERROR_FEEDBACK_H