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
CompressorRegistry::Register reg(
    "vanilla_error_feedback",
    [](const kwargs_t& kwargs) -> std::unique_ptr<BaseCompressor> {
      auto kwargs_clone = kwargs;
      kwargs_clone.erase("error_feedback_type");
      auto compressor_ptr = CompressorRegistry::Create(kwargs_clone);
      BPS_LOG(DEBUG) << "Register Error feedback "
                     << "compressor_type=" << kwargs_clone["compressor_type"];
      return std::unique_ptr<VanillaErrorFeedbackCompressor>(
          new VanillaErrorFeedbackCompressor(std::move(compressor_ptr)));
    });
}

VanillaErrorFeedbackCompressor::VanillaErrorFeedbackCompressor(
    std::unique_ptr<BaseCompressor> compressor_ptr)
    : ErrorFeedback(std::move(compressor_ptr)) {}

VanillaErrorFeedbackCompressor::~VanillaErrorFeedbackCompressor() = default;

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
void VanillaErrorFeedbackCompressor::UpdateGradient(ByteBuf grad, int dtype) {
  this->_cpu_reducer->sum(grad.data, _error.get(), grad.size,
                          static_cast<DataType>(dtype));
}
#else
// server version decompressor
void VanillaErrorFeedbackCompressor::UpdateGradient(ByteBuf grad, int dtype) {
  float len = grad.size / getDataTypeLength(dtype);
  this->_cpu_reducer->sum(grad.data, _error.get(), grad.size,
                          static_cast<DataType>(dtype), 1.0 / len);
}
#endif

void VanillaErrorFeedbackCompressor::UpdateError(ByteBuf corrected, int dtype,
                                                 ByteBuf* compressed) {
  // TODO: we may remove this copy in the futher
  this->_cpu_reducer->copy(_error.get(), compressed->data, compressed->size);
  ByteBuf compressed_copy{_error.get(), compressed->size},
      decompressed{_debuf.get(), corrected.size};
  Decompress(compressed_copy, dtype, &decompressed);
  this->_cpu_reducer->sum(_error.get(), corrected.data, decompressed.data,
                          corrected.size, static_cast<DataType>(dtype), -1.0);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps