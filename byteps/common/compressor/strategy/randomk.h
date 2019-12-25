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

#ifndef BYTEPS_COMPRESS_STRAT_RANDOMK_H
#define BYTEPS_COMPRESS_STRAT_RANDOMK_H

#include "compressor/base_compressor.h"

namespace byteps {
namespace common {
namespace compress {

/**
 * \brief TODO
 */
class RandomkCompressor : public BaseCompressor {
 public:
  explicit RandomkCompressor(int k);
  ~RandomkCompressor();

  TensorType Compress(const TensorType& grad) override;

  TensorType Decompress(const TensorType& compressed_grad) override;

 private:
  int _k;
};
}
}
}

#endif  // BYTEPS_COMPRESS_STRAT_RANDOMK_H