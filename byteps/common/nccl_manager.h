// Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_NCCL_MANAGER_H
#define BYTEPS_NCCL_MANAGER_H

#include <vector>
#include <memory>
#include <queue>
#include "common.h"
#include "scheduled_queue.h"
#include "communicator.h"

namespace byteps {
namespace common {


struct NcclGroupEntry {
    cudaEvent_t cuda_event;
    std::vector<std::shared_ptr<TensorTableEntry>> tasks;
    std::vector<BytePSScheduledQueue*> queues;
};


class NcclManager {

public:

    NcclManager(std::shared_ptr<BytePSComm> comm);
    ~NcclManager() {
        if (_nccl_stream) {
            CUDA_CALL(cudaStreamDestroy(*_nccl_stream));
        }
    }

    cudaStream_t* GetNcclStream() { return _nccl_stream; }
    ncclComm_t* GetNcclComm() { return &_nccl_comm; }
    int GetNcclGroupSize() { return _nccl_group_size; }

    void EnqueueNcclGroup(std::shared_ptr<NcclGroupEntry> e);
    std::shared_ptr<NcclGroupEntry> DequeueNcclGroup();

private:

    void InitGlobalEnv();

    cudaStream_t* _nccl_stream;
    ncclUniqueId* _nccl_id;
    ncclComm_t _nccl_comm;
    
    // global user-defined env
    int _nccl_group_size;

    // for pipelining nccl
    std::mutex _nccl_mutex;
    std::queue<std::shared_ptr<NcclGroupEntry>> _nccl_pipeline;

};


} // namespace common
} // namespace byteps

#endif // BYTEPS_NCCL_MANAGER_H