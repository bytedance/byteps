// Copyright 2020 Amazon Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_COMPRESSOR_IMPL_SPARSE_ERROR_FEEDBACK_H
#define BYTEPS_COMPRESSOR_IMPL_SPARSE_ERROR_FEEDBACK_H

#include "../error_feedback.h"
#include "../utils.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief Sprase Error Feedback Compressor
 *
 * Sparse Error Feedback Update.
 *
 * Error-correction: error needs to be scaled with \frac{\eta_{t-1}}{\eta_t}.
 */
class SparseErrorFeedbackCompressor : public ErrorFeedback {
 public:
  SparseErrorFeedbackCompressor(size_t size, DataType dtype,
                                std::unique_ptr<Compressor> cptr, size_t k,
                                unsigned int seed = 0);
  virtual ~SparseErrorFeedbackCompressor();

 protected:
  void UpdateGradient(tensor_t grad) override;

  void UpdateError(tensor_t corrected, tensor_t compressed) override;

 private:
  /*!
   * \brief learning rate
   *
   * read from file each step
   */
  double _pre_lr, _cur_lr;

  int _fd;
  void* _mm;

  size_t _k;
  XorShift128PlusBitShifterRNG _rng;
  std::vector<uint32_t> _selected_idx;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_VANILLA_ERROR_FEEDBACK_H