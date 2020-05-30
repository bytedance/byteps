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

void ErrorFeedback::Init(size_t aligned_size) {
  _compressor_ptr->Init(aligned_size);
  _error.reset(new char[aligned_size]);
  memset(_error.get(), 0, aligned_size);
  _cpu_reducer.reset(new CpuReducer(nullptr));
}

void ErrorFeedback::Compress(tensor_t grad, tensor_t& compressed) {
  // before: grad += error
  UpdateGradient(grad);

  // TODO: look strange
  compressed.data = _error.get();
  // compress
  _compressor_ptr->Compress(grad, compressed);

  UpdateError(grad, compressed);
}

void ErrorFeedback::Decompress(tensor_t compressed, tensor_t& decompressed) {
  _compressor_ptr->Decompress(compressed, decompressed);
}

void ErrorFeedback::UpdateError(tensor_t corrected, tensor_t compressed) {
  _compressor_ptr->FastUpdateError({_error.get()}, corrected, compressed);
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps