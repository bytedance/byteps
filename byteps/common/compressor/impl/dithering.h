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

#ifndef BYTEPS_COMPRESSOR_IMPL_MULTIBIT_H
#define BYTEPS_COMPRESSOR_IMPL_MULTIBIT_H

#include "../compressor.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief TODO
 */
class DitheringCompressor : public Compressor {
 public:
  DitheringCompressor(size_t size, DataType dtype, int k)
      : Compressor(size, dtype), _k(k){};
  virtual ~DitheringCompressor() = default;

  tensor_t Compress(tensor_t grad) override;

  tensor_t Decompress(tensor_t compressed) override;

 private:
  int _k;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_MULTIBIT_H