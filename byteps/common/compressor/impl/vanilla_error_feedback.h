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

#ifndef BYTEPS_COMPRESSOR_IMPL_VANILLA_ERROR_FEEDBACK_H
#define BYTEPS_COMPRESSOR_IMPL_VANILLA_ERROR_FEEDBACK_H

#include "../error_feedback.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief Vanilla Error Feedback Compressor
 *
 * paper: Communication-efficient distributed blockwise momentum sgd with
 * error-feedback
 * https://arxiv.org/pdf/1905.10936.pdf
 *
 * each worker i:
 *    p_{t,i} <- g_{t,i} + \frac{\eta_{t-1}}{\eta_t} e_{t,i}
 *    c_{t,i} <- Compress(p_{t,i})
 *    e_{t,i} <- p_{t,i} - c_{t,i}
 *
 * server:
 *    \tilde{p}_{t} <- \frac{1}{M} \sum_{i=1}^{M} c_{t,i}
 * +\frac{\eta_{t-1}}{\eta_{t}} \tilde{e_t} \tilde{e}_{t+1} <-
 * \tilde{p}_{t}-\tilde{c_t}
 *
 * Error-correction: error needs to be scaled with \frac{\eta_{t-1}}{\eta_t}.
 */
class VanillaErrorFeedbackCompressor : public ErrorFeedback {
 public:
  VanillaErrorFeedbackCompressor(size_t size, DataType dtype,
                                 std::unique_ptr<Compressor> cptr);
  virtual ~VanillaErrorFeedbackCompressor();

 protected:
  void UpdateGradient(tensor_t grad) override;

 private:
  /*!
   * \brief learning rate
   *
   * read from file each step
   */
  double _pre_lr, _cur_lr;

  int _fd;
  void* _mm;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_VANILLA_ERROR_FEEDBACK_H