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

#include "multibit.h"

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorFactory::Register reg("multibit", [](const CompressorParam&
                                                   param) -> CompressorPtr {
  auto iter = param.find("compressor_k");
  if (iter == param.end()) {
    BPS_LOG(DEBUG) << "Multibit Compressor needs parameter \"compressor_k\"";
    return nullptr;
  }
  int k = std::stoi(iter->second);
  return std::unique_ptr<BaseCompressor>(new MultibitCompressor(k));
});
}

MultibitCompressor::MultibitCompressor(int k) : _k(k){};

MultibitCompressor::~MultibitCompressor() = default;

TensorType MultibitCompressor::Compress(const TensorType& grad) {
  // TODO
}

TensorType MultibitCompressor::Decompress(const TensorType& compressed_grad) {
  // TODO
}
}
}
}