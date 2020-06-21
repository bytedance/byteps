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
#include "local_comm.h"
#include "dist_comm.h"

namespace byteps {
namespace sparse {

enum OP { GATHER, SCATTER };

static std::vector<void*> _embedBuffers;
static std::vector<void*> _denseBuffers;

// length of the _denseBuffers
static size_t _denseBufferLen;

// length of _embedBuffers, per local GPU (size: # GPUs)
static std::vector<size_t> _localEmbedBufLens; 

// embed buf len of each global gpu (dim0: num_worker, dim1: localsize) 
static std::vector<std::vector<size_t>> _globalEmbedBufLens; 

// sum of the embed buf len of each worker (size: # workers)
static std::vector<size_t> _globalTotalEmbedBufLens;

// local communication handler
static std::vector<std::unique_ptr<LocalGatherComm>> _local_gather_comms;
static std::vector<std::unique_ptr<LocalScatterComm>> _local_scatter_comms;

// distributed communication handler
static std::vector<std::unique_ptr<DistGatherComm>> _dist_gather_comms;
static std::vector<std::unique_ptr<DistScatterComm>> _dist_scatter_comms;

// The following are extern APIs
extern "C" void bytepsSparseInit(std::vector<void*>& embedBuffers, std::vector<void*>& denseBuffers, std::vector<size_t>& embedBufferLens, size_t denseBufferLen);
extern "C" void bytepsSparseShutdown();
extern "C" void bytepsGatherExecAsync(int local_rank, cudaStream_t stream);
extern "C" void bytepsScatterExecAsync(int local_rank, cudaStream_t stream);
extern "C" void bytepsSynchronize(int local_rank, cudaStream_t stream, OP op);

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_H
