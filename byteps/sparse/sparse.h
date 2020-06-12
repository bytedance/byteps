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
#include "loop.h"
#include "cpu_reducer.h"

namespace byteps {
namespace sparse {

enum OP { GATHER, SCATTER };

static std::vector<void*> _embedBuffers;
static std::vector<void*> _denseBuffers; // For storing gathered embedding vectors for the dense layer to use.
static std::vector<std::vector<size_t>> _embedBufferLens; 
static std::vector<void*> _cpuBuffers;
static size_t _denseBufferLen;
static std::vector<std::unique_ptr<LocalGatherComm>>  _local_gather_comms;
static std::vector<std::unique_ptr<LocalScatterComm>>  _local_scatter_comms;

// Buffers for dense layers when calling DenseReduceAsync
static std::vector<void*> _denseDeltaBeforeReduceBuffers;   // In GPU
static std::vector<void*> _denseDeltaAfterReduceBuffers;    // In GPU
static void* _cpuDenseDeltaBuffers;
static void* _cpuDenseLatestBuffers;

static size_t _denseDeltaBufferLength;  // Unit in bytes.

static QueueExecLoop* _denseReduceLoop;
static ::byteps::common::CpuReducer* _denseReducer;

// The following are extern APIs
extern "C" void bytepsSparseInit(std::vector<void*>& embedBuffers,
                                 std::vector<void*>& denseBuffers,
                                 std::vector<int>& embedBufferLens,
                                 int size);
extern "C" void bytepsSparseInitDense(std::vector<void*>& denseDeltaBeforeReduceBuffers,
                                      std::vector<void*>& denseDeltaAfterReduceBuffers,
                                      int sizeDenseDelta);

// This is not thread safe! The caller must make sure it's not called concurrently and from rank 0 to num_gpu-1.
extern "C" void bytepsSparseInitDensePerGPU(int device_id /* starts with 0 */,
                                            void* denseDeltaBeforeReduceBuffer,
                                            void* denseDeltaAfterReduceBuffer,
                                            int sizeDenseDelta);

extern "C" void bytepsSparseShutdown();
extern "C" void bytepsGatherExecAsync(int local_rank, cudaStream_t stream);
extern "C" void bytepsScatterExecAsync(int local_rank, cudaStream_t stream);
extern "C" void bytepsDenseReduceExecAsync(int local_rank, cudaStream_t stream);
extern "C" void bytepsSynchronize(int local_rank, cudaStream_t stream, OP op);

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_H
