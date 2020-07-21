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

std::unique_ptr<Compressor> CompressorRegistry::Create(const kwargs_t& kwargs,
                                                       size_t size, DataType dtype) {
#ifndef BYTEPS_BUILDING_SERVER
  const std::string types[] = {"momentum_type", "ef_type", "compressor_type"};
#else
  // server do not need momentum
  const std::string types[] = {"ef_type", "compressor_type"};
#endif
  for (auto& type : types) {
    auto iter = kwargs.find(type);
    if (iter != kwargs.end()) {
      auto ctor = CompressorRegistry::Find(iter->second + "_" + type);
      return ctor(kwargs, size, dtype);
    }
  }

  return nullptr;
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps