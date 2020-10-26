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

void ErrorFeedback::Compress(tensor_t grad, tensor_t& output) {
  BPS_CHECK(grad.data);

  // 1. p <- g + e
  UpdateGradient(grad);

  tensor_t error{_buf.get(), _size, _dtype};

  // 2. c <- Compress(p) 3. e <- p - c
  _cptr->FusedCompress(grad, output, error);

#ifndef BYTEPS_BUILDING_SERVER
  auto ptr = reinterpret_cast<float*>(error.data);
  double scale = 0.0;
  size_t len = error.size / sizeof(float);
#pragma omp parallel for simd reduction(max : scale)
  for (size_t i = 0; i < len; i++) {
    scale = scale > std::abs(ptr[i]) ? scale : std::abs(ptr[i]);
  }

  BPS_LOG(INFO) << "error's max norm=" << scale << " size=" << error.size;
#endif
}

void ErrorFeedback::Decompress(tensor_t compressed, tensor_t& output) {
  // directly forward to internal compressor
  _cptr->Decompress(compressed, output);
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps