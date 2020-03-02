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

#include "randomk.h"

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "randomk", [](const kwargs_t& kwargs) -> std::unique_ptr<BaseCompressor> {
      auto iter = kwargs.find("compressor_k");
      if (iter == kwargs.end()) {
        BPS_LOG(WARNING)
            << "Randomk Compressor needs parameter \"compressor_k\"";
        return nullptr;
      }
      int k = std::stoi(iter->second);
      BPS_LOG(DEBUG) << "Register Randomk Compressor "
                     << "k=" << k;
      return std::unique_ptr<BaseCompressor>(new RandomkCompressor(k));
    });
}

RandomkCompressor::RandomkCompressor(int k) : _k(k){};

RandomkCompressor::~RandomkCompressor() = default;

void RandomkCompressor::Compress(ByteBuf grad, int dtype, ByteBuf& compressed) {
  // TODO
}

void RandomkCompressor::Decompress(ByteBuf compressed, int dtype,
                                   ByteBuf& decompressed) {
  // TODO
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps