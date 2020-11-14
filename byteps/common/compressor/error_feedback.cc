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

tensor_t ErrorFeedback::Compress(tensor_t grad) {
  // 1. grad <- grad + error
  UpdateGradient(grad);

  // 2. c <- Compress(grad)
  auto compressed = _cptr->Compress(grad);

  // 3. e <- grad - Decompress(c)
  UpdateError(grad, compressed);

  return compressed;
}

tensor_t ErrorFeedback::Decompress(tensor_t compressed) {
  // directly forward to internal compressor
  return _cptr->Decompress(compressed);
}

void ErrorFeedback::UpdateError(tensor_t corrected, tensor_t compressed) {
  tensor_t error{_error.get(), _size, corrected.dtype};
  _cptr->FastUpdateError(error, corrected, compressed);
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps