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
    // init and sycn NCCL-reduce-id using out-of-band socket
    _nccl_id = (ncclUniqueId*) malloc(sizeof(ncclUniqueId));
    if (BytePSGlobal::IsRootDevice()) { // only root create nccl id

        NCCLCHECK(ncclGetUniqueId(_nccl_id));

        // the log is just for debug, the actual length of nccl id is 128
        BPS_LOG(DEBUG) << "root nccl_reduce_id is " << (*(long long int*)_nccl_id);

        comm->broadcastSignal(BytePSGlobal::GetLocalRank(), _nccl_id, sizeof(ncclUniqueId));

    } else {
        int src;
        int rc = comm->recvSignal(&src, _nccl_id, sizeof(ncclUniqueId));

        BPS_CHECK_EQ(rc, sizeof(ncclUniqueId)) << rc << ", " << sizeof(ncclUniqueId);

        BPS_LOG(DEBUG) << "recv nccl_reduce_id is " << (*(long long int*)_nccl_id)
                       << ", local_rank=" << BytePSGlobal::GetLocalRank();
    }

    _nccl_stream  = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    int greatest_priority;
    CUDA_CALL(cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    CUDA_CALL(cudaStreamCreateWithPriority(_nccl_stream, cudaStreamNonBlocking, greatest_priority));
    CUDA_CALL(cudaStreamSynchronize(*_nccl_stream));
    //initializing NCCL rank
    NCCLCHECK(ncclCommInitRank(&_nccl_comm, BytePSGlobal::GetLocalSize(), *_nccl_id, BytePSGlobal::GetLocalRank()));
    return;
}

void NcclManager::InitGlobalEnv() { // init all global env/param here
    _nccl_group_size = (getenv("BYTEPS_NCCL_GROUP_SIZE") ? atoi(getenv("BYTEPS_NCCL_GROUP_SIZE")) : 4);
    BPS_LOG(DEBUG) << "nccl_group_size" << " set to " << _nccl_group_size;
    return;
}

void NcclManager::EnqueueNcclGroup(std::shared_ptr<NcclGroupEntry> e) {
    std::lock_guard<std::mutex> lock(_nccl_mutex);
    _nccl_pipeline.push(e);
    return;
}

std::shared_ptr<NcclGroupEntry> NcclManager::DequeueNcclGroup() {
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
