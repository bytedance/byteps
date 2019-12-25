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

#include "compressor/base_compressor.h"

namespace byteps {
namespace common {
namespace compress {

/**
 *  \brief Error feedback Decorator
 *
 *  extend error feedback behavior to any compressor at run-time
 */
class ErrorFeedback : public BaseCompressor {
 public:
  explicit ErrorFeedback(std::unique_ptr<BaseCompressor> compressor_ptr);
  ~ErrorFeedback();

  /**
   * \brief initiaze `_compressor_ptr` buffer first
   */
  void InitBuff(size_t len) override;

  /**
   * \brief compress with error feedback
   *
   *  invoke `UpdateGradient` and `UpdateError` before and after
   *  `_compressor_ptr->Compress(*)`
   */
  TensorType Compress(const TensorType& grad);

  /**
   * \brief directly forward to `_compressor_ptr->Decompress(*)`
   */
  TensorType Decompress(const TensorType& compressed_grad);

 protected:
  /**
   * \brief correct gradient with error
   *
   * grad += error
   *
   * \param grad input gradient to be updated
   * \return corrected gradient
   */
  virtual TensorType UpdateGradient(const TensorType& grad) = 0;

  /**
   * \brief update error
   *
   * error = grad - decompress(compressed_corrected_grad)
   *
   * \param grad original gradient (uncompressed)
   */
  virtual void UpdateError(const TensorType& grad) = 0;

 private:
  /**
   * \brief compressor to be extended
   */
  std::unique_ptr<BaseCompressor> _compressor_ptr;

  char* _decompress_buff;
  char* _error_buff;
};
}
}
}

#endif  // BYTEPS_COMPRESS_ERROR_FEEDBACK_H