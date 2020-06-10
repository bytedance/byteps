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


#ifndef BYTEPS_SPARSE_H
#define BYTEPS_SPARSE_H

#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <memory>
#include <string>
#include "gossip/include/gossip.cuh"
#include "gossip/include/cudahelpers/cuda_helpers.cuh"
#include "gossip/include/plan_parser.hpp"
#include "gossip/include/clipp/include/clipp.h"

#include "common.h"
#include "util.h"

namespace byteps {
namespace sparse {

static std::vector<void*> _embedBuffers;
static std::vector<void*> _denseBuffers;
static std::vector<std::vector<size_t>> _bufferLengths; 
static std::vector<std::vector<size_t>> _offsets;
static ps::KVWorker<char>* _ps;
static std::vector<void*> _cpuBuffers;
static size_t _denseBufferLength;

// Local gossip communication (gather)
static std::unique_ptr<gossip::context_t> gather_cxt_;
static std::unique_ptr<gossip::gather_t> gather_;
static std::vector<float*> srcs_gather_;
static std::vector<size_t> lens_gather_;
static std::vector<std::vector<size_t>> table_gather_;

// The following are extern APIs
extern "C" void bytepsSparseInit(std::vector<void*>& embedBuffers, std::vector<void*>& denseBuffers, std::vector<int>& bufferLengths, int size);
extern "C" void bytepsSparseShutdown();
extern "C" void bytepsGather(int local_rank, cudaStream_t stream);
extern "C" void bytepsScatter(int local_rank, cudaStream_t stream);

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_H
