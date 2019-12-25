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

#include "compressor/error_feedback.h"

namespace byteps {
namespace common {
namespace compress {

ErrorFeedback::ErrorFeedback(std::unique_ptr<BaseCompressor> compressor_ptr)
    : _compressor_ptr(std::move(compressor_ptr)) {}

ErrorFeedback::~ErrorFeedback() {
  if (_decompress_buff) delete[] _decompress_buff;
  if (_error_buff) delete[] _error_buff;
}

void ErrorFeedback::InitBuff(size_t len) {
  _compressor_ptr->InitBuff(len);
  _decompress_buff = new char[len];
  _error_buff = new char[len];
}

TensorType ErrorFeedback::Compress(const TensorType& grad) {
  // before: grad += error
  auto corrected_grad = UpdateGradient(grad);
  // compress
  auto compressed_grad = _compressor_ptr->Compress(corrected_grad);
  // after: error = grad - decompress(compressed_corrected_grad)
  UpdateError(grad);

  return compressed_grad;
}

TensorType ErrorFeedback::Decompress(const TensorType& compressed_grad) {
  return _compressor_ptr->Decompress(compressed_grad);
}
}
}
}