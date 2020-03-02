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

#ifndef BYTEPS_COMPRESS_VANILLA_EF_H
#define BYTEPS_COMPRESS_VANILLA_EF_H

#include "../error_feedback.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief TODO
 */
class VanillaErrorFeedbackCompressor : public ErrorFeedback {
 public:
  explicit VanillaErrorFeedbackCompressor(
      std::unique_ptr<BaseCompressor> compressor_ptr);
  virtual ~VanillaErrorFeedbackCompressor();

 protected:
  void UpdateGradient(ByteBuf grad, int dtype) override;

  void UpdateError(ByteBuf corrected, int dtype, ByteBuf& compressed) override;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESS_VANILLA_EF_H