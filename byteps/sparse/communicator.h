// Copyright 2020 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_SPARSE_COMMUNICATOR_H
#define BYTEPS_SPARSE_COMMUNICATOR_H

#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <memory>
#include <string>
#include <mutex>
#include "gossip/include/gossip.cuh"
#include "gossip/include/cudahelpers/cuda_helpers.cuh"
#include "gossip/include/plan_parser.hpp"
#include "ps/ps.h"

namespace byteps {
namespace sparse {

class SparseComm {
 public:
  virtual void ExecAsync() = 0;
  virtual void Sync() = 0;
}; // class SparseComm

} // namespace sparse
} // namespace byteps

#endif  // BYTEPS_SPARSE_COMMUNICATOR_H
