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

CompressorPtr CompressorFactory::create(const CompressorParam& param_dict) const {
  auto param_iter = param_dict.find("compressor_type");
  if (param_iter == param_dict.end()) {
    return nullptr;
  }

  auto& name = *param_iter;
  auto func_iter = _create_funcs.find(name);
  if (func_iter == _create_funcs.end()) {
    BPS_LOG(ERROR) << "No compressor registered under name:" << name;
    return nullptr;
  }

  param_iter = param_dict.find("error_feedback_type");
  if (param_iter == param_dict.end()) {
    return func_iter->second(param_dict);
  }

  func_iter = _create_funcs.find(param_iter->second + "_error_feedback");
  if (func_iter == _create_funcs.end()) {
    BPS_LOG(ERROR) << "No compressor with error feedback registered under name:" << name;
    return nullptr;
  }

  return func_iter->second(param_dict);
}

BaseCompressor::BaseCompressor() = default;

BaseCompressor::~BaseCompressor() {
  if (_compress_buff) delete[] _compress_buff;
}

void BaseCompressor::InitBuff(size_t len) { _compress_buff = new char[len]; }
}
}
}