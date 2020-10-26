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

#include "compressor_registry.h"

#include <vector>

namespace byteps {
namespace common {
namespace compressor {

CompressorRegistry::map_t CompressorRegistry::_ctor_map;

CompressorRegistry::Register::Register(std::string name, ctor_t ctor) {
  BPS_CHECK_EQ(_ctor_map.count(name), 0)
      << "Duplicate registration of compressor under name " << name;
  _ctor_map.emplace(name + "_type", std::move(ctor));
  BPS_LOG(INFO) << name << " compressor is registered";
}

CompressorRegistry::ctor_t CompressorRegistry::Find(const std::string& name) {
  auto it = _ctor_map.find(name);
  if (it == _ctor_map.end()) {
    BPS_LOG(FATAL) << "No compressor registered under name:" << name;
  }
  return it->second;
}

std::unique_ptr<Compressor> CompressorRegistry::Create(kwargs_t kwargs,
                                                       size_t size,
                                                       DataType dtype) {
#ifndef BYTEPS_BUILDING_SERVER
  std::vector<std::string> types = {
      "compressor_type",
      "ef_type",
      "momentum_type",
  };
  // lower data type should be cast into fp32
  if (dtype == BYTEPS_FLOAT16) {
    if (kwargs.find("mixed_precision") != kwargs.end() &&
        kwargs["mixed_precision"] == "true") {
      kwargs["cast_type"] = "fp16";
    }
  }
#else
  // server do not need momentum and cast
  std::vector<std::string> types = {
      "compressor_type",
      "ef_type",
  };
#endif
  std::unique_ptr<Compressor> internal_cptr = nullptr;
  for (auto& type : types) {
    auto iter = kwargs.find(type);
    if (iter != kwargs.end()) {
      auto ctor = CompressorRegistry::Find(iter->second + "_" + type);
      internal_cptr =
          std::move(ctor(kwargs, size, dtype, std::move(internal_cptr)));
    }
  }

  return internal_cptr;
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps