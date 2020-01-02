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

#include "compressor/base_compressor.h"

#include "logging.h"

namespace byteps {
namespace common {
namespace compressor {

CompressorFactory& CompressorFactory::instance() {
  static CompressorFactory factory;
  return factory;
}

CompressorFactory::CompressorFactory() = default;

CompressorFactory::~CompressorFactory() = default;

CompressorFactory::Register::Register(std::string name,
                                       CreateFunc create_func) {
  auto& factory = CompressorFactory::instance();
  auto iter = factory._create_funcs.find(name);
  if (iter == factory._create_funcs.end()) {
    factory._create_funcs.emplace(name, create_func);
  } else {
    BPS_LOG(FATAL) << "Duplicate registration of compressor under name "
                   << name;
  }
}

CompressorPtr CompressorFactory::create(std::string name,
                                         const CompressorParam& param) const {
  auto iter = _create_funcs.find(name);
  if (iter == _create_funcs.end()) {
    BPS_LOG(ERROR) << "No compressor registered under name:" << name;
    return nullptr;
  }
  return iter->second(param);
}

BaseCompressor::BaseCompressor() = default;

BaseCompressor::~BaseCompressor() {
  if (_compress_buff) delete[] _compress_buff;
}

void BaseCompressor::InitBuff(size_t len) { _compress_buff = new char[len]; }
}
}
}