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

#include <memory>
#include <chrono>
#include <cuda_runtime.h>

#include "logging.h"
#include "core_loops.h"
#include "common.h"
#include "global.h"

namespace byteps {
namespace common {

void FinishOrProceed(std::shared_ptr<TensorTableEntry> task) {
    auto &queue_list = task->queue_list;
    BPS_CHECK_GE(queue_list.size(), 1);
    auto this_op = queue_list[0];
    queue_list.erase(queue_list.begin());
    if (queue_list.size() > 0) {
        BPS_LOG(TRACE) << "Rank=" << BytePSGlobal::GetRank()
                       << " finishes " << LogStrings[this_op] << ", tensor: " << task->tensor_name
                       << ", key=" << task->key << "; Passing to the next queue.";
        BytePSGlobal::GetScheduledQueue(queue_list[0])->addTask(task);
    } else {
        BPS_CHECK(task->counter_ptr) << task->tensor_name << " counter_ptr is null";
        int v = task->counter_ptr.get()->fetch_add(1);
        if (v == (int)(task->total_partnum-1)) {
            BPS_LOG(TRACE) << "Rank=" << BytePSGlobal::GetRank()
                           << "Finish processing tensor: " << task->tensor_name;
            task->callback(Status::OK());
        }
    }
    return;
}

bool RunCoordinateLoopOnce(QueueType this_op) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask();
    if (task){
        int rank = BytePSGlobal::GetLocalRank();
        int key  = task->key;

        // first send to next queue and then broadcast signal
        // to guarantee the entry is available when getTask(key) at Reduce/Broadcast thread
        FinishOrProceed(task);

        BytePSCommSignal sig;
        std::shared_ptr<BytePSComm> comm;

        switch (this_op) {
            case COORDINATE_REDUCE: {
                sig = REDUCE_READY;
                comm = BytePSGlobal::GetNccl()->GetSignalComm();
                break;
            }
            case COORDINATE_BROADCAST: {
                sig = BCAST_READY;
                comm = BytePSGlobal::GetNccl()->GetSignalComm();
                break;
            }
            case COORDINATE_PUSH: {
                sig = PUSH_READY;
                comm = BytePSGlobal::GetBasicComm();
                break;
            }
            default:
                BPS_CHECK(0) << "unsupported op: " << this_op;
        }

        BPS_CHECK_NE(rank, comm->getRoot()) << "only non-root device should enter COORDINATE loop";

        struct BytePSCommMsg msg = { rank, sig, key };
        comm->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));

        BPS_LOG(TRACE) << task->tensor_name << " send coordinate info: "
                       << "Signal=" << sig
                       << ", rank=" << rank
                       << ", key="  << key;

        q->reportFinish(task->len);

    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

inline void PostNcclCalls(std::shared_ptr<byteps::common::TensorTableEntry> task, QueueType this_op) {

    BPS_CHECK(this_op == REDUCE || this_op == BROADCAST) << "Only REDUCE and BROADCAST use NCCL.";
    auto tensor = (this_op == REDUCE) ? task->tensor : task->output;
    BPS_CHECK(tensor);
    BPS_CHECK(tensor->data());
    BPS_CHECK_EQ(0, tensor->size() % tensor->shape().num_elements());

    int key  = task->key;
    auto len = task->len;
    auto offset = task->offset;
    auto unit_len = tensor->size() / tensor->shape().num_elements();
    auto p = (char*)(tensor->data()) + offset;
    auto nccl_dtype = getNcclDataType(tensor->dtype());

    auto nccl = BytePSGlobal::GetNccl();
    auto nccl_stream = nccl->GetStream(key, this_op);
    auto nccl_comm = nccl->GetComm(key, this_op);
    auto nccl_root = nccl->GetRoot(key, this_op);
    auto nccl_size = nccl->GetSize();
    auto nccl_rank = nccl->GetRank(key, this_op);

    auto num_elem_per_gpu = len / nccl_size / unit_len;
    auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

    BPS_LOG(TRACE) << task->tensor_name << " calling NCCL "
                    << LogStrings[this_op]
                    << " (rank=" << nccl_rank
                    << ") key=" << key
                    << ", elements=" << len/unit_len
                    << ", device=" << task->device;

    if (this_op == REDUCE) {
        if (num_elem_per_gpu) {
            NCCLCHECK(ncclReduceScatter((const void*) p,
                                    (void*) (p + nccl_rank * num_elem_per_gpu * unit_len),
                                    (size_t) num_elem_per_gpu,
                                    (ncclDataType_t) nccl_dtype,
                                    (ncclRedOp_t) ncclSum,
                                    (ncclComm_t) nccl_comm,
                                    (cudaStream_t) nccl_stream));
        }
        if (left_elem) {
            NCCLCHECK(ncclReduce((const void*) (p + len - left_elem * unit_len),
                                    (void*) (p + len - left_elem * unit_len),
                                    (size_t) left_elem,
                                    (ncclDataType_t) nccl_dtype,
                                    (ncclRedOp_t) ncclSum,
                                    (int) nccl_root,
                                    (ncclComm_t) nccl_comm,
                                    (cudaStream_t) nccl_stream));
        }
    }
    else {
        if (num_elem_per_gpu) {
            NCCLCHECK(ncclAllGather((const void*) (p + nccl_rank * num_elem_per_gpu * unit_len),
                                    (void*) p,
                                    (size_t) num_elem_per_gpu,
                                    (ncclDataType_t) nccl_dtype,
                                    (ncclComm_t) nccl_comm,
                                    (cudaStream_t) nccl_stream));
        }
        if (left_elem) {
            NCCLCHECK(ncclBroadcast((const void*) (p + len - left_elem * unit_len),
                                    (void*) (p + len - left_elem * unit_len),
                                    (size_t) left_elem,
                                    (ncclDataType_t) nccl_dtype,
                                    (int) nccl_root,
                                    (ncclComm_t) nccl_comm,
                                    (cudaStream_t) nccl_stream));
        }

    }
}


bool RunRootNcclLoopOnce() {
    auto signal_comm = BytePSGlobal::GetNccl()->GetSignalComm();
    int root = signal_comm->getRoot();
    int rank = BytePSGlobal::GetLocalRank();
    BPS_CHECK_EQ(rank, root);

    int nccl_size = BytePSGlobal::GetNccl()->GetSize();
    QueueType nccl_ops[] = { REDUCE, BROADCAST };

    auto nccl_entry = std::make_shared<NcclGroupEntry>(); 
    auto &tasks = nccl_entry->tasks;
    auto &queues = nccl_entry->queues;

    NCCLCHECK(ncclGroupStart());
    for (auto this_op : nccl_ops) {
        auto q = BytePSGlobal::GetScheduledQueue(this_op);
        for (int i = 0; i < BytePSGlobal::GetNccl()->GetGroupSize(); i++) {
            auto task = q->getTask();
            if (!task) { break; }
            tasks.push_back(task);
            queues.push_back(q);
            
            if (task->device != CPU_DEVICE_ID) { // GPU
                if (nccl_size > 1) {
                    // notify non-root devices
                    struct BytePSCommMsg msg = { rank,
                                                 (this_op == REDUCE) ? DO_REDUCE : DO_BROADCAST,
                                                 (int)(task->key) };
                    signal_comm->broadcastSignal(&msg,
                                                 sizeof(BytePSCommMsg));
                    PostNcclCalls(task, this_op);
                }
            }
        }
    }
    if (tasks.size()) {
        struct BytePSCommMsg msg = { rank, DO_GROUP, 0 };
        signal_comm->broadcastSignal(&msg, sizeof(BytePSCommMsg));
        NCCLCHECK(ncclGroupEnd());
        nccl_entry->RecordEvents();
        BPS_LOG(TRACE) << "NCCL Group size=" << tasks.size() << " rank=" << rank;
        BytePSGlobal::GetNccl()->EnqueueGroup(nccl_entry);
    }
    else {
        NCCLCHECK(ncclGroupEnd());
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunNonRootNcclLoopOnce() {
    auto signal_comm = BytePSGlobal::GetNccl()->GetSignalComm();
    int root = signal_comm->getRoot();
    int rank = BytePSGlobal::GetLocalRank();
    BPS_CHECK_NE(rank, root);

    auto nccl_entry = std::make_shared<NcclGroupEntry>(); 
    auto &tasks = nccl_entry->tasks;
    auto &queues = nccl_entry->queues;
    struct BytePSCommMsg msg = {};

    NCCLCHECK(ncclGroupStart());
    while (1) {
        signal_comm->recvSignalFromRoot(&msg, sizeof(BytePSCommMsg));
        if (msg.signal == DO_GROUP) { break; }
        QueueType this_op = REDUCE;
        if (msg.signal == DO_BROADCAST) {
            this_op = BROADCAST;
        }
        else {
            BPS_CHECK_EQ(msg.signal, DO_REDUCE) << msg.signal << ", " << DO_REDUCE;
        }

        int key = msg.key;

        auto q = BytePSGlobal::GetScheduledQueue(this_op);
        auto task = q->getTask(key);
        BPS_CHECK(task);

        tasks.push_back(task);
        queues.push_back(q);

        if (task->device != CPU_DEVICE_ID) { // GPU
            PostNcclCalls(task, this_op);
        }
    }
    NCCLCHECK(ncclGroupEnd());

    nccl_entry->RecordEvents();
    BytePSGlobal::GetNccl()->EnqueueGroup(nccl_entry);
    return true;
}

bool RunSyncNcclOnce() {
    auto nccl_entry = BytePSGlobal::GetNccl()->DequeueGroup();
    if (nccl_entry) {
        nccl_entry->SynchronizeEvents();
        for (size_t i = 0; i < nccl_entry->tasks.size(); i++) {
            FinishOrProceed(nccl_entry->tasks[i]);
            // Only root manages credits
            if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
                nccl_entry->queues[i]->reportFinish(nccl_entry->tasks[i]->len);
            }
        }
        nccl_entry->DestroyEvents();
        BPS_LOG(TRACE) << "Finished NCCL Group size=" << nccl_entry->tasks.size()
                       << " rank=" << BytePSGlobal::GetLocalRank();
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

bool RunCopyDevice2HostLoopOnce() {
    QueueType this_op = COPYD2H;
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask();

    if (task) {
        auto copy_d2h_Stream =  BytePSGlobal::GetCopyDevice2HostStream();
        auto tensor = task->tensor;
        BPS_CHECK(tensor);
        int key  = task->key;

        auto nccl = BytePSGlobal::GetNccl();
        auto nccl_root = nccl->GetRoot(key, REDUCE);
        auto nccl_size = nccl->GetSize();
        auto nccl_rank = nccl->GetRank(key, REDUCE);

        if (task->device != CPU_DEVICE_ID) { // GPU
            auto len = task->len;
            auto offset = task->offset;
            auto p = (char*)(tensor->data()) + offset;
            auto unit_len = tensor->size() / tensor->shape().num_elements();
            char* cpubuff;
            if (BytePSGlobal::IsCrossPcieSwitch()) {
                BPS_CHECK(task->pcie_cpubuff.size());
                cpubuff = (char*)(task->pcie_cpubuff[BytePSGlobal::GetPcieSwitchIndex()]) + offset;
            }
            else {
                cpubuff = (char*)(task->cpubuff) + offset;
            }

            BPS_CHECK(cpubuff) << task->tensor_name
                               << ": CPU buffer not initialized, size=" << len;

            auto num_elem_per_gpu = len / nccl_size / unit_len;
            auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

            auto copy_len = num_elem_per_gpu * unit_len;
            if (left_elem && nccl_root == nccl_rank) {
                copy_len += left_elem * unit_len;
            }

            CUDA_CALL(cudaMemcpyAsync((void *) (cpubuff + nccl_rank * num_elem_per_gpu * unit_len),
                                      (const void *) (p + nccl_rank * num_elem_per_gpu * unit_len),
                                      (size_t) copy_len,
                                      (cudaMemcpyKind) cudaMemcpyDeviceToHost,
                                      (cudaStream_t) *copy_d2h_Stream));
            CUDA_CALL(cudaStreamSynchronize(*copy_d2h_Stream));
        }

        FinishOrProceed(task);
        q->reportFinish(task->len);
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

bool RunPcieReduceLoopOnce() {
    BPS_CHECK(BytePSGlobal::IsCrossPcieSwitch());
    QueueType this_op = PCIE_REDUCE;
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask();
    if (task) {
        auto reducer = BytePSGlobal::GetCpuReducer();
        if (!reducer->isRoot()) {
            // send signal to root
            int rank = BytePSGlobal::GetLocalRank();
            int key  = task->key;
            BytePSCommSignal sig = PCIE_REDUCE_READY;
            struct BytePSCommMsg msg = { rank, sig, key };
            reducer->getComm()->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));
        }
        else {
            if (task->device != CPU_DEVICE_ID) { // GPU
                auto tensor = task->tensor;
                
                int key  = task->key;
                auto len = task->len;
                auto offset = task->offset;
                auto unit_len = tensor->size() / tensor->shape().num_elements();

                auto nccl = BytePSGlobal::GetNccl();
                auto nccl_root = nccl->GetRoot(key, REDUCE);
                auto nccl_size = nccl->GetSize();
                auto nccl_rank = nccl->GetRank(key, REDUCE);

                auto num_elem_per_gpu = len / nccl_size / unit_len;
                auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

                auto copy_len = num_elem_per_gpu * unit_len;
                if (left_elem && nccl_root == nccl_rank) {
                    copy_len += left_elem * unit_len;
                }

                auto total_offset = offset + nccl_rank * num_elem_per_gpu * unit_len;

                reducer->sum((void*)((char*)(task->cpubuff) + total_offset),
                             (void*)((char*)(task->pcie_cpubuff[0]) + total_offset),
                             copy_len, task->tensor->dtype());
            }
        }

        FinishOrProceed(task);
        q->reportFinish(task->len);
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

bool RunPushLoopOnce() {
    QueueType this_op = PUSH;
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask();
    if (task) {
        BPS_CHECK(BytePSGlobal::IsRootDevice()) << "only root device should enter PUSH loop";
        // TODO: allow merging
        auto offset = task->offset;
        auto len = task->len;

        char* data;
        if (task->device != CPU_DEVICE_ID) {
            BPS_CHECK(task->cpubuff);
            data = const_cast<char*>(static_cast<const char*>(task->cpubuff) + offset);
        } else {
            BPS_CHECK(task->tensor);
            data = const_cast<char*>(static_cast<const char*>(task->tensor->data()) + offset);
        }

        // get metadata
        const int dtype = task->tensor->dtype();

        // false means not to delete data when SArray is deleted
        ps::SArray<char> vals(data, len, false);

        int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
        auto& pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
        BytePSGlobal::GetPS()->ZPush(
            pskv.keys, vals, pskv.lens, cmd,
            [task, q]() {
                FinishOrProceed(task);
                q->reportFinish(task->len);
            }
        );
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

bool RunPullLoopOnce() {
    QueueType this_op = PULL;
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask();
    if (task) {
        BPS_CHECK(BytePSGlobal::IsRootDevice()) << "only root device should enter PULL loop";
        // TODO: allow merging
        auto offset = task->offset;
        auto len = task->len;

        char* data;
        if (task->device != CPU_DEVICE_ID) { // GPU
            BPS_CHECK(task->cpubuff);
            data = const_cast<char*>(static_cast<const char*>(task->cpubuff) + offset);
        } else { // CPU
            BPS_CHECK(task->output);
            data = const_cast<char*>(static_cast<const char*>(task->output->data()) + offset);
        }

        // get metadata
        const int dtype = task->output->dtype();

        // false means not to delete data when SArray is deleted
        auto vals = new ps::SArray<char>(data, len, false);

        int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
        auto& pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
        // issue pull
        BytePSGlobal::GetPS()->ZPull(
            pskv.keys, vals, &pskv.lens, cmd,
            [vals, task, q]() {
                delete vals;
                FinishOrProceed(task);
                q->reportFinish(task->len);
            });
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

void CopyHost2Device(std::shared_ptr<byteps::common::TensorTableEntry> task) {
    auto copy_h2d_stream = BytePSGlobal::GetCopyHost2DeviceStream();    
    auto tensor = task->output;
    BPS_CHECK(tensor);
    int key  = task->key;
    auto nccl = BytePSGlobal::GetNccl();
    auto nccl_root = nccl->GetRoot(key, BROADCAST);
    auto nccl_size = nccl->GetSize();
    auto nccl_rank = nccl->GetRank(key, BROADCAST);
    auto name = task->tensor_name;
    auto len = task->len;
    auto offset = task->offset;
    auto cpubuff = (char*)(task->cpubuff) + offset;
    BPS_CHECK(cpubuff) << name << ": CPU buffer not initialized, size=" << len;

    char* gpu_addr = const_cast<char*>(static_cast<const char*>(tensor->data()) + offset);
    auto unit_len = tensor->size() / tensor->shape().num_elements();
    auto num_elem_per_gpu = len / nccl_size / unit_len;
    auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);
    
    auto copy_len = num_elem_per_gpu * unit_len;
    if (left_elem && nccl_root == nccl_rank) {
        copy_len += left_elem * unit_len;
    }

    CUDA_CALL(cudaMemcpyAsync((void *) (gpu_addr + nccl_rank * num_elem_per_gpu * unit_len),
                                (const void *) (cpubuff + nccl_rank * num_elem_per_gpu * unit_len),
                                (size_t) copy_len,
                                (cudaMemcpyKind) cudaMemcpyHostToDevice,
                                (cudaStream_t) *copy_h2d_stream));
    CUDA_CALL(cudaStreamSynchronize(*copy_h2d_stream));
    return;
}

bool RunRootCopyHost2DeviceLoopOnce() {
    QueueType this_op = COPYH2D;
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask();

    if (task) {
        int key  = task->key;
        int local_rank = BytePSGlobal::GetLocalRank();
        int local_size = BytePSGlobal::GetLocalSize();

        if (task->device != CPU_DEVICE_ID) { // GPU
            if (local_size > 1) {
                // notify non-root devices
                struct BytePSCommMsg msg = { local_rank,
                                             DO_COPYH2D,
                                             key };
                BytePSGlobal::GetBasicComm()->broadcastSignal(&msg,
                                                            sizeof(BytePSCommMsg));
            }
            CopyHost2Device(task);
        }

        FinishOrProceed(task);
        q->reportFinish(task->len);
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

bool RunNonRootCopyListenLoopOnce() {
    auto signal_comm = BytePSGlobal::GetBasicComm();
    int root = signal_comm->getRoot();
    int rank = BytePSGlobal::GetLocalRank();
    BPS_CHECK_NE(root, rank);

    struct BytePSCommMsg msg = {};

    signal_comm->recvSignalFromRoot(&msg, sizeof(BytePSCommMsg));
    BPS_CHECK_EQ(msg.signal, DO_COPYH2D) << msg.signal;

    BytePSGlobal::GetCopyTable()->AddReadyCount(msg.key);

    BPS_LOG(TRACE) << "NonRootCopyListenLoop recved from root"
                       << ", signal=" << msg.signal
                       << ", key="    << msg.key
                       << ", myrank=" << rank;
    return true;
}

bool RunNonRootCopyHost2DeviceLoopOnce() {
    QueueType this_op = COPYH2D;
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask();

    if (task) {
        if (task->device != CPU_DEVICE_ID) { // GPU
            CopyHost2Device(task);
        }
        FinishOrProceed(task);
        q->reportFinish(task->len);
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

void CoordinateReduceLoop() {
    while (RunCoordinateLoopOnce(COORDINATE_REDUCE) && !BytePSGlobal::ShouldShutdown()) {}
}

void CoordinateBroadcastLoop() {
    while (RunCoordinateLoopOnce(COORDINATE_BROADCAST) && !BytePSGlobal::ShouldShutdown()) {}
}

void CoordinatePushLoop() {
    while (RunCoordinateLoopOnce(COORDINATE_PUSH) && !BytePSGlobal::ShouldShutdown()) {}
}

void PcieReduceLoop() {
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
    while (RunPcieReduceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void RootNcclLoop() {
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
    while (RunRootNcclLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void NonRootNcclLoop() {
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
    while (RunNonRootNcclLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void SyncNcclLoop() {
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
    while (RunSyncNcclOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void CopyDevice2HostLoop() {
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
    while (RunCopyDevice2HostLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void PushLoop() {
    while (RunPushLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void PullLoop() {
    while (RunPullLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void RootCopyHost2DeviceLoop() {
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
    while (RunRootCopyHost2DeviceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void NonRootCopyListenLoop() {
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
    while (RunNonRootCopyListenLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void NonRootCopyHost2DeviceLoop() {
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
    while (RunNonRootCopyHost2DeviceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}


} // namespace common
} // namespace byteps
