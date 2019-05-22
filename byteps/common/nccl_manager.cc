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

#include "nccl_manager.h"
#include "logging.h"
#include "global.h"

namespace byteps {
namespace common {

NcclManager::NcclManager(std::shared_ptr<BytePSComm> comm) {
    InitGlobalEnv();
    ConstructRings();

    // init and sycn NCCL-reduce-id using out-of-band socket
    _nccl_id = (ncclUniqueId*) malloc(sizeof(ncclUniqueId) * _nccl_pcie_size);
    _nccl_comm = (ncclComm_t*) malloc(sizeof(ncclComm_t) * _nccl_pcie_size);
    for (size_t i = 0; i < _nccl_pcie_size; i++) {
        auto nccl_id = _nccl_id + i;
        auto nccl_comm = _nccl_comm + i;
        if (BytePSGlobal::IsRootDevice()) { // only root create nccl id
            NCCLCHECK(ncclGetUniqueId(nccl_id));
            // the log is just for debug, the actual length of nccl id is 128
            BPS_LOG(DEBUG) << "root nccl_reduce_id is " << (*(long long int*)nccl_id);
            comm->broadcastSignal(BytePSGlobal::GetLocalRank(), nccl_id, sizeof(ncclUniqueId));

        }
        else {
            int src;
            int rc = comm->recvSignal(&src, nccl_id, sizeof(ncclUniqueId));
            BPS_CHECK_EQ(rc, sizeof(ncclUniqueId)) << rc << ", " << sizeof(ncclUniqueId);
            BPS_LOG(DEBUG) << "recv nccl_reduce_id is " << (*(long long int*)nccl_id)
                        << ", local_rank=" << BytePSGlobal::GetLocalRank();
        }

        //initializing NCCL rank
        auto it = std::find(_rings[i].begin(), _rings[i].end(), BytePSGlobal::GetLocalRank());
        auto rank = std::distance(_rings[i].begin(), it);
        NCCLCHECK(ncclCommInitRank(nccl_comm, BytePSGlobal::GetLocalSize(), *nccl_id, rank));
    }

    _nccl_stream  = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    int greatest_priority;
    CUDA_CALL(cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    CUDA_CALL(cudaStreamCreateWithPriority(_nccl_stream, cudaStreamNonBlocking, greatest_priority));
    CUDA_CALL(cudaStreamSynchronize(*_nccl_stream));
    return;
}

ncclComm_t NcclManager::GetComm(int key) {
    return _nccl_comm[key % _nccl_pcie_size];
}

int NcclManager::GetRoot(int key, QueueType op) {
    int comm_index = key % _nccl_pcie_size;
    int pcie_index = key % (_nccl_pcie_size * _nccl_pcie_num) / _nccl_pcie_size;
    int root_index = -1;
    if (op == REDUCE) {
        root_index = (pcie_index + 1) * _nccl_pcie_size - 1;
    }
    else {
        BPS_CHECK_EQ(op, BROADCAST) << "Unknown OP for NcclManager.";
        root_index = pcie_index * _nccl_pcie_size;
    }
    BPS_CHECK_GT(root_index, -1);
    return _rings[comm_index][root_index];
}

// Example: 8 GPUs, 4 per PCIe switch
//
// 4 NCCL communicators
// 0 1 2 3 | 4 5 6 7
// 1 2 3 0 | 5 6 7 4
// 2 3 0 1 | 6 7 4 5
// 3 0 1 2 | 7 4 5 6
//
// reduce
// 1st ring, 0->1->2->3->cpubuff->4->5->6->7->cpubuff, 4->5->6->7->cpubuff->0->1->2->3->cpubuff
// 2nd ring, 1->2->3->0->cpubuff->5->6->7->4->cpubuff, 5->6->7->4->cpubuff->1->2->3->0->cpubuff
// 3rd ring, 2->3->0->1->cpubuff->6->7->4->5->cpubuff, 6->7->4->5->cpubuff->2->3->0->1->cpubuff
// 4th ring, 3->0->1->2->cpubuff->7->4->5->6->cpubuff, 7->4->5->6->cpubuff->3->0->1->2->cpubuff      
//
// broadcast
// 1st ring, cpubuff->0->1->2->3->cpubuff->4->5->6->7, cpubuff->4->5->6->7->cpubuff->0->1->2->3
// 2nd ring, cpubuff->1->2->3->0->cpubuff->5->6->7->4, cpubuff->5->6->7->4->cpubuff->1->2->3->0
// 3rd ring, cpubuff->2->3->0->1->cpubuff->6->7->4->5, cpubuff->6->7->4->5->cpubuff->2->3->0->1
// 4th ring, cpubuff->3->0->1->2->cpubuff->7->4->5->6, cpubuff->7->4->5->6->cpubuff->3->0->1->2
//
void NcclManager::ConstructRings() {
    BPS_LOG(DEBUG) << "Constructing NCCL communicators.";
    for (size_t i = 0; i < _nccl_pcie_size; i++) {
        _rings.push_back(std::vector<int>());
        std::string log("");
        for (size_t j = 0; j < _nccl_pcie_num; j++) {
            for (size_t k = 0; k < _nccl_pcie_size; k++) {
                int rank = (k + i) % _nccl_pcie_size + j * _nccl_pcie_size;
                _rings[i].push_back(rank);
                log = log + std::to_string(rank) + ' ';
            }
        }
        BPS_LOG(DEBUG) << log;
    }
    return;
}

void NcclManager::InitGlobalEnv() { // init all global env/param here
    _nccl_group_size = (getenv("BYTEPS_NCCL_GROUP_SIZE") ?
                        atoi(getenv("BYTEPS_NCCL_GROUP_SIZE")) : 4);
    BPS_LOG(DEBUG) << "nccl_group_size" << " set to " << _nccl_group_size;
    
    _nccl_pcie_size = (getenv("BYTEPS_PCIE_SWITCH_SIZE") ?
                       atoi(getenv("BYTEPS_PCIE_SWITCH_SIZE")) : 4);
    auto local_size = BytePSGlobal::GetLocalSize();
    _nccl_pcie_num = local_size / _nccl_pcie_size;
    if (!_nccl_pcie_num) {
        _nccl_pcie_size = local_size;
        _nccl_pcie_num = 1;
    }
    else {
        BPS_CHECK_EQ(local_size % _nccl_pcie_size, 0)
                     << "BytePS does not support unbalanced PCIe switches.";
    }

    BPS_LOG(DEBUG) << "nccl_pcie_size" << " set to " << _nccl_pcie_size;
    BPS_LOG(DEBUG) << "nccl_pcie_num" << " set to " << _nccl_pcie_num;
    
    return;
}

void NcclManager::EnqueueGroup(std::shared_ptr<NcclGroupEntry> e) {
    std::lock_guard<std::mutex> lock(_nccl_mutex);
    _nccl_pipeline.push(e);
    return;
}

std::shared_ptr<NcclGroupEntry> NcclManager::DequeueGroup() {
    std::lock_guard<std::mutex> lock(_nccl_mutex);
    if (!_nccl_pipeline.size()) {
        return nullptr;
    }
    auto r = _nccl_pipeline.front();
    _nccl_pipeline.pop();
    return r;
}

} // namespace common
} // namespace byteps
