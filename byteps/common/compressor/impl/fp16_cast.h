// Copyright 2020 Amazon Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_COMPRESSOR_IMPL_FP16_CAST_H
#define BYTEPS_COMPRESSOR_IMPL_FP16_CAST_H

#include "../cast.h"

namespace byteps {
namespace common {
namespace compressor {
/*!
 * \brief FP16 Cast Wrapper
 *
 * Cast fp16 gradients into fp32 and then compress
 *
 * \sa Cast
 */
class FP16CastCompressor : public Cast {
 public:
  FP16CastCompressor(size_t size, DataType dtype,
                     std::unique_ptr<Compressor> cptr)
      : Cast(size, dtype, std::move(cptr)) {}
  ~FP16CastCompressor() override = default;

 protected:
  tensor_t CastToFP32(tensor_t grad) override;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_FP16_CAST_H