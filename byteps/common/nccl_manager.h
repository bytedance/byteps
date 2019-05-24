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


class NcclGroupEntry {

public:

    void RecordEvents();
    void SynchronizeEvents();
    void DestroyEvents();

    std::vector<std::shared_ptr<TensorTableEntry>> tasks;
    std::vector<BytePSScheduledQueue*> queues;

private:

    std::vector<cudaEvent_t> _events;
};


class NcclManager {

public:

    NcclManager(std::shared_ptr<BytePSComm> comm);
    ~NcclManager() {
        if (_nccl_stream) {
            CUDA_CALL(cudaStreamDestroy(*_nccl_stream));
        }
    }

    int GetGroupSize() { return _nccl_group_size; }
    void EnqueueGroup(std::shared_ptr<NcclGroupEntry> e);
    std::shared_ptr<NcclGroupEntry> DequeueGroup();

    cudaStream_t GetStream(int key, QueueType op);
    ncclComm_t GetComm(int key, QueueType op);
    int GetRoot(int key, QueueType op);

private:

    void InitGlobalEnv();
    void ConstructRings();

    cudaStream_t* _nccl_stream;
    ncclUniqueId* _nccl_id;
    ncclComm_t* _nccl_comm;
    
    // global user-defined env
    size_t _nccl_group_size;
    size_t _nccl_pcie_size;
    size_t _nccl_pcie_num;

    // for pipelining nccl
    std::mutex _nccl_mutex;
    std::queue<std::shared_ptr<NcclGroupEntry>> _nccl_pipeline;

    // for multi-ring
    std::vector<std::vector<int>> _rings;

    std::shared_ptr<BytePSComm> _signal_comm;

};


} // namespace common
} // namespace byteps

#endif // BYTEPS_NCCL_MANAGER_H