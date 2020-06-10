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

#include "util.h"
#include "common.h"
#include "communicator.h"

namespace byteps {
namespace sparse {

static std::vector<void*> _embedBuffers;
static std::vector<void*> _denseBuffers;
static std::vector<std::vector<size_t>> _embedBufferLens; 
static ps::KVWorker<char>* _ps;
static std::vector<void*> _cpuBuffers;
static size_t _denseBufferLength;
static std::vector<std::unique_ptr<LocalGatherComm>>  _local_gather_comms;

// The following are extern APIs
extern "C" void bytepsSparseInit(std::vector<void*>& embedBuffers, std::vector<void*>& denseBuffers, std::vector<int>& embedBufferLens, int size);
extern "C" void bytepsSparseShutdown();
extern "C" void bytepsGather(int local_rank, cudaStream_t stream);
extern "C" void bytepsScatter(int local_rank, cudaStream_t stream);

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_H
