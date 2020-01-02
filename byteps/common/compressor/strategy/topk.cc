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

#include "compressor/strategy/topk.h"

#include "logging.h"

namespace byteps {
namespace common {
namespace compressor {

CompressorFactory::Register reg(
    "topk", [](const CompressorParam& param) -> CompressorPtr {
      auto iter = param.find("compressor_k");
      if (iter == param.end()) {
        BPS_LOG(FATAL) << "Topk Compressor needs parameter \"compressor_k\"";
        return nullptr;
      }
      int k = std::stoi(iter->second);
      return std::unique_ptr<BaseCompressor>(new TopkCompressor(k));
    });

TopkCompressor::TopkCompressor(int k) : _k(k){};

TopkCompressor::~TopkCompressor() = default;

TensorType TopkCompressor::Compress(const TensorType& grad) {
  // TODO
}

TensorType TopkCompressor::Decompress(const TensorType& compressed_grad) {
  // TODO
}
}
}
}