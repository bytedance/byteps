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

#ifndef BYTEPS_COMPRESSOR_COMPRESSOR_REGISTRY_H
#define BYTEPS_COMPRESSOR_COMPRESSOR_REGISTRY_H

#include "compressor.h"
#include "utils.h"

namespace byteps {
namespace common {
namespace compressor {

class CompressorRegistry {
 public:
  // constructor of compressor
  using ctor_t = std::function<std::unique_ptr<Compressor>(
      const kwargs_t& kwargs, size_t size, DataType dtype)>;

  using map_t = std::unordered_map<std::string, ctor_t>;

  struct Register {
    Register(std::string name, ctor_t ctor);
  };

  static ctor_t Find(const std::string& name);

  static std::unique_ptr<Compressor> Create(const kwargs_t& kwargs, size_t size,
                                            DataType dtype);

 private:
  static map_t _ctor_map;

  CompressorRegistry() = delete;
  ~CompressorRegistry() = delete;
};

}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_COMPRESSOR_REGISTRY_H