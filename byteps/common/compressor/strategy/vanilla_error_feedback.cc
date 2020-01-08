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

#include "vanilla_error_feedback.h"

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorFactory::Register reg(
    "vanilla_error_feedback",
    [](const CompressorParam& param) -> CompressorPtr {
      auto param_copy = param;
      param_copy.erase("error_feedback_type");
      auto& factory = CompressorFactory::instance();
      auto compressor_ptr = factory.create(param_copy);
      return std::unique_ptr<VanillaErrorFeedbackCompressor>(
          new VanillaErrorFeedbackCompressor(std::move(compressor_ptr)));
    });
}

VanillaErrorFeedbackCompressor::VanillaErrorFeedbackCompressor(
    std::unique_ptr<BaseCompressor> compressor_ptr)
    : ErrorFeedback(std::move(compressor_ptr)) {}

VanillaErrorFeedbackCompressor::~VanillaErrorFeedbackCompressor() = default;

TensorType VanillaErrorFeedbackCompressor::UpdateGradient(
    const TensorType& grad) {
  // TODO
}

void VanillaErrorFeedbackCompressor::UpdateError(const TensorType& grad) {
  // TODO
}
}
}
}