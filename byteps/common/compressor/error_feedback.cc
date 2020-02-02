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

#include "error_feedback.h"

namespace byteps {
namespace common {
namespace compressor {

ErrorFeedback::ErrorFeedback(std::unique_ptr<BaseCompressor> compressor_ptr)
    : _compressor_ptr(std::move(compressor_ptr)) {}

ErrorFeedback::~ErrorFeedback() = default;

void ErrorFeedback::AllocateBuffer(size_t len) {
  _compressor_ptr->AllocateBuffer(len);
  _decode_buf.reset(new char[len]);
  _error_buf.reset(new char[len]);
}

ByteBuf ErrorFeedback::Compress(const ByteBuf& grad) {
  // before: grad += error
  auto corrected_grad = UpdateGradient(grad);
  // compress
  auto compressed_grad = _compressor_ptr->Compress(corrected_grad);
  // after: error = corrected_grad - decompress(compressed_corrected_grad)
  UpdateError(corrected_grad, compressed_grad);

  return compressed_grad;
}

ByteBuf ErrorFeedback::Decompress(const ByteBuf& compressed_grad) {
  return _compressor_ptr->Decompress(compressed_grad);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps