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

#include <unistd.h>

#include "../compressor_registry.h"
#include "test_error_feedback.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "test_ef",
    [](const kwargs_t& kwargs, size_t size,
       DataType dtype) -> std::unique_ptr<Compressor> {
      // register cptr
      auto kwargs_clone = kwargs;
      kwargs_clone.erase("ef_type");
      auto cptr = CompressorRegistry::Create(kwargs_clone, size, dtype);
      BPS_CHECK_NE(cptr, nullptr);
      return std::unique_ptr<TestErrorFeedbackCompressor>(
          new TestErrorFeedbackCompressor(size, dtype, std::move(cptr)));
    });
}

TestErrorFeedbackCompressor::TestErrorFeedbackCompressor(
    size_t size, DataType dtype, std::unique_ptr<Compressor> cptr)
    : ErrorFeedback(size, dtype, std::move(cptr)) {}

void TestErrorFeedbackCompressor::UpdateGradient(tensor_t grad) {
  this->_cpu_reducer->sum(grad.data, _error.get(), grad.size,
                          static_cast<DataType>(grad.dtype));
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps