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

#include <cstring>
#include <memory>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

#include "operations.h"
#include "global.h"

namespace byteps {
namespace common {

bool RunReduceLoopOnce() {
    QueueType this_op = REDUCE;
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto reduce_stream =  BytePSGlobal::GetReduceStream();
    if (q->pendingSize() > 0) {
        auto task = q->getTask();
        BPS_CHECK(task->tensor);

        if (task->device != CPU_DEVICE_ID) { // GPU
            auto name = task->tensor_name;
            auto len = task->len;
            auto offset = task->offset;
            auto cpubuff = task->cpubuff + offset;
            BPS_CHECK(cpubuff) << name << ": CPU buffer not initialized, size=" << len;
            CUDA_CALL(cudaMemcpyAsync(cpubuff, task->tensor->data() + offset, len, cudaMemcpyDeviceToHost, *reduce_stream));
            CUDA_CALL(cudaStreamSynchronize(*reduce_stream));
        }

        if (task->last_op != this_op) { // TODO: should check the boundary
            BPS_LOG(TRACE) << "Finish reducing tensor: " << task->tensor_name
                           << ", passing it to the next queue " << this_op+1;
            BytePSGlobal::GetScheduledQueue(static_cast<QueueType>(this_op+1))->addTask(task);
        } else {
            task->callback(Status::OK());
        }
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

bool RunPushLoopOnce() {
    QueueType this_op = PUSH;
    auto q = BytePSGlobal::GetScheduledQueue(PUSH);
    if (q->pendingSize() > 0) {
        // TODO: allow merging
        auto task = q->getTask();
        auto offset = task->offset;
        auto len = task->len;

        char* data;
        if (task->device != CPU_DEVICE_ID) {
            BPS_CHECK(task->cpubuff + offset);
            data = const_cast<char*> (static_cast<const char*> (task->cpubuff + offset));
        } else {
            BPS_CHECK(task->tensor);
            data = const_cast<char*> (static_cast<const char*> (task->tensor->data() + offset));
        }

        // get metadata
        const int dtype = task->tensor->dtype();

        // false means not to delete data when SArray is deleted
        ps::SArray<char> vals(data, len, false);

        int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
        auto& pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
        BytePSGlobal::GetPS()->ZPush(
            pskv.keys, vals, pskv.lens, cmd,
            [task, this_op]() {
                if (task->last_op != this_op) {
                    BPS_LOG(TRACE) << "Finish pushing tensor: " << task->tensor_name
                                   << ", passing it to the next queue " << this_op+1;
                    BytePSGlobal::GetScheduledQueue(static_cast<QueueType>(this_op+1))->addTask(task);
                } else {
                    BPS_CHECK(task->counter_ptr) << task->tensor_name << " counter_ptr is null";
                    int v = task->counter_ptr.get()->fetch_add(1);
                    if (v == (task->total_partnum-1)) {
                        BPS_LOG(TRACE) << "Finish pushing tensor: " << task->tensor_name
                                   << ", invoking callback.";
                        task->callback(Status::OK());
                    }
                }
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
    auto q = BytePSGlobal::GetScheduledQueue(PULL);
    if (q->pendingSize() > 0) {
        // TODO: allow merging
        auto task = q->getTask();
        auto offset = task->offset;
        auto len = task->len;

        char* data;
        if (task->device != CPU_DEVICE_ID) { // GPU
            BPS_CHECK(task->cpubuff);
            data = const_cast<char*> (static_cast<const char*> (task->cpubuff + offset));
        } else { // CPU
            BPS_CHECK(task->output);
            data = const_cast<char*> (static_cast<const char*> (task->output->data() + offset));
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
            [vals, task, this_op]() {
                delete vals;
                if (task->last_op != this_op) {
                    BPS_LOG(TRACE) << "Finish pulling tensor: " << task->tensor_name
                                   << ", passing it the next queue " << this_op+1;
                    BytePSGlobal::GetScheduledQueue(static_cast<QueueType>(this_op+1))->addTask(task);
                } else {
                    task->callback(Status::OK());
                }
            });
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

bool RunBroadcastLoopOnce() {
    QueueType this_op = BROADCAST;
    auto q = BytePSGlobal::GetScheduledQueue(BROADCAST);
    auto broadcast_stream = BytePSGlobal::GetBroadcastStream();
    if (q->pendingSize() > 0) {
        auto task = q->getTask();
        BPS_CHECK(task->output);

        if (task->device != CPU_DEVICE_ID) { // GPU
            auto name = task->tensor_name;
            auto len = task->len;
            auto offset = task->offset;

            auto cpubuff = task->cpubuff + offset;
            BPS_CHECK(cpubuff) << name << ": CPU buffer not initialized, size=" << len;
            char* gpu_addr = const_cast<char*> (static_cast<const char*> (task->output->data() + offset));
            CUDA_CALL(cudaMemcpyAsync(gpu_addr, cpubuff, len, cudaMemcpyHostToDevice, *broadcast_stream));
            CUDA_CALL(cudaStreamSynchronize(*broadcast_stream));
        }

        BPS_CHECK_EQ(this_op, QueueNum-1) << "this should the last op";
        BPS_CHECK(task->counter_ptr) << task->tensor_name << " counter_ptr is null";
        int v = task->counter_ptr.get()->fetch_add(1);
        if (v == (task->total_partnum-1)) {
            BPS_LOG(TRACE) << "Finish broadcasting tensor: " << task->tensor_name;
            task->callback(Status::OK());
        }
    }
    else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    return true;
}

void ReduceLoop() {
    while (RunReduceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void PushLoop() {
    while (RunPushLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void PullLoop() {
    while (RunPullLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

void BroadcastLoop() {
    while (RunBroadcastLoopOnce() && !BytePSGlobal::ShouldShutdown()) {}
}

extern "C" {

void byteps_init() {
    BytePSGlobal::Init();
    LoopFunction func[ThreadNum] = {ReduceLoop, PushLoop, PullLoop, BroadcastLoop};
    BytePSGlobal::Start(func);
    return;
}

void byteps_shutdown() {
    BytePSGlobal::Shutdown();
    BPS_LOG(TRACE) << "BytePS is shutdown.";
    return;
}

int byteps_rank() {
    return BytePSGlobal::GetRank();
}

int byteps_local_rank() {
    return BytePSGlobal::GetLocalRank();
}

int byteps_size() {
    return BytePSGlobal::GetSize();
}

int byteps_local_size() {
    return BytePSGlobal::GetLocalSize();
}

} // extern "C"

Status CheckInitialized() {
    return BytePSGlobal::CheckInit();
}

void PartitionTensor(std::shared_ptr<TensorTableEntry> entry,
                    std::vector<std::shared_ptr<TensorTableEntry> > &partitions,
                    bool is_reduce_push) {
    BPS_CHECK(entry->counter_ptr) << entry->tensor_name << " counter pointer is null";
    auto size = is_reduce_push ? entry->tensor->size() : entry->output->size();
    auto bound = BytePSGlobal::_partition_bound;
    auto accumulated = 0;
    int i = 0;

    while (accumulated < size) {
        std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
        // will assign the key later, so don't do it now
        // e->key = entry->key;
        e->tensor_name = entry->tensor_name + std::string("_") + std::to_string(i);
        e->context = entry->context;
        e->ready_event = entry->ready_event;
        e->device = entry->device;
        e->priority = entry->priority;
        e->version = entry->version;
        e->callback = entry->callback;
        e->cpubuff = entry->cpubuff;
        e->last_op = entry->last_op;
        e->tensor = is_reduce_push ? BPS_CHECK_NOTNULL(entry->tensor) : NULL;
        e->output = is_reduce_push ? NULL : BPS_CHECK_NOTNULL(entry->output);
        e->offset = accumulated;
        e->len = ((size - accumulated) > bound) ? bound : (size - accumulated);
        e->counter_ptr = entry->counter_ptr;
        e->total_partnum = entry->total_partnum;

        accumulated += e->len;
        ++i;

        partitions.push_back(e);
    }
}

Status EnqueueTensorPush(BPSContext &context,
                        std::shared_ptr<Tensor> input,
                        std::shared_ptr<ReadyEvent> ready_event,
                        const std::string &name,
                        const int device, const int priority, const int version,
                        StatusCallback callback, QueueType last_op) {
    BPS_CHECK(input) << name << " tensor is null";

    std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
    e->tensor_name = name;
    e->context = &context;
    e->tensor = input;
    e->output = NULL;
    e->ready_event = ready_event;
    e->device = device;
    e->priority = priority;
    e->version = version;
    e->callback = callback;
    e->cpubuff = context.cpubuff;
    e->last_op = last_op;
    e->counter_ptr = std::make_shared<std::atomic_int>(0);
    e->total_partnum = context.key_list.size();

    std::vector<std::shared_ptr<TensorTableEntry> > partitions;
    PartitionTensor(e, partitions, true);
    BPS_CHECK_EQ(context.key_list.size(), partitions.size());

    unsigned int accumulated = 0;
    for (unsigned int i = 0; i < partitions.size(); ++i) {
        auto task = partitions[i];
        task->key = context.key_list[i]; // assign the key now
        BPS_LOG(TRACE) << "EnqueueTensorPush: " << task->tensor_name
                       << ", key=" << task->key
                       << ", offset=" << task->offset
                       << ", len=" << task->len
                       << ", device=" << task->device
                       << ", last_op=" << task->last_op;
        BytePSGlobal::GetScheduledQueue(REDUCE)->addTask(task);
        accumulated += task->len;
    }
    BPS_CHECK_EQ(accumulated, e->tensor->size()) << "accumulated partition size not equal to original tensor size";

    return Status::OK();
}

Status EnqueueTensorPull(BPSContext &context,
                        std::shared_ptr<Tensor> output,
                        std::shared_ptr<ReadyEvent> ready_event,
                        const std::string &name,
                        const int device, const int priority, const int version,
                        StatusCallback callback, QueueType last_op) {
    BPS_CHECK(output) << name << " tensor is null";

    std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
    e->tensor_name = name;
    e->context = &context;
    e->tensor = NULL;
    e->output = output;
    e->ready_event = ready_event;
    e->device = device;
    e->priority = priority;
    e->version = version;
    e->callback = callback;
    e->cpubuff = context.cpubuff;
    e->last_op = last_op;
    e->counter_ptr = std::make_shared<std::atomic_int>(0);
    e->total_partnum = context.key_list.size();

    std::vector<std::shared_ptr<TensorTableEntry> > partitions;
    PartitionTensor(e, partitions, false);
    BPS_CHECK_EQ(context.key_list.size(), partitions.size());

    unsigned int accumulated = 0;
    for (unsigned int i = 0; i < partitions.size(); ++i) {
        auto task = partitions[i];
        task->key = context.key_list[i]; // assign the key now
        BPS_LOG(TRACE) << "EnqueueTensorPull: " << task->tensor_name
                       << ", key=" << task->key
                       << ", offset=" << task->offset
                       << ", len=" << task->len
                       << ", device=" << task->device
                       << ", last_op=" << task->last_op;
        BytePSGlobal::GetScheduledQueue(PULL)->addTask(task);
        accumulated += task->len;
    }
    BPS_CHECK_EQ(accumulated, e->output->size()) << "accumulated partition size not equal to original tensor size";

    return Status::OK();
}

Status InitTensor(BPSContext &context,
                  std::shared_ptr<Tensor> tensor,
                  std::shared_ptr<ReadyEvent> ready_event,
                  const std::string &name, const int device,
                  StatusCallback callback) {

    // Only rank 0 pushes the initialization
    if (BytePSGlobal::GetRank() == 0) {
        auto key_list = context.key_list;
        BPS_LOG(TRACE) << "Init " << name
                       << ", size=" << tensor->size()
                       << ", parts=" << key_list.size()
                       << ", device=" << device;
        // get metadata
        size_t size = tensor->size();
        const int dtype = tensor->dtype();

        char* data;
        if (device != CPU_DEVICE_ID) { // GPU
            BPS_CHECK_EQ(size, context.buff_len);
            BPS_CHECK(context.cpubuff);
            CUDA_CALL(cudaMemcpy(context.cpubuff, tensor->data(), size, cudaMemcpyDeviceToHost));
            data = const_cast<char*> (static_cast<const char*> (context.cpubuff));
        } else { // CPU
            data = const_cast<char*> (static_cast<const char*> (tensor->data()));
        }

        auto bound = BytePSGlobal::_partition_bound;
        unsigned int accumulated = 0;
        auto i = 0;
        BPS_CHECK_EQ(key_list.size(), (unsigned int) (size+bound-1)/bound) // round up
                       << key_list.size()
                       << ", size=" << size
                       << ", bound=" << bound;
        while (accumulated < size) {
            auto key = key_list[i];
            int len = ((size - accumulated) > bound) ? bound : (size - accumulated);

            // false means not to delete data when SArray is deleted
            ps::SArray<char> vals(data + accumulated, len, false);

            int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);

            auto& pskv = BytePSGlobal::EncodeDefaultKey(key, len);
            BytePSGlobal::GetPS()->Wait(BytePSGlobal::GetPS()->ZPush(
                pskv.keys, vals, pskv.lens, cmd));

            accumulated += len;
            ++i;
        }
        BPS_CHECK_EQ(accumulated, size);
        BPS_CHECK_EQ((unsigned int) i, key_list.size());
    }

    ps::Postoffice::Get()->Barrier(0, ps::kWorkerGroup);

    callback(Status::OK());
    return Status::OK();
}

BPSContext& GetContextFromName(const std::string &name) {
    return BytePSGlobal::GetContextFromName(name);
}

bool IsTensorInitialized(const std::string &name, size_t size, int device, DataType dtype) {
    return BytePSGlobal::IsTensorInitialized(name, size, device, dtype);
}

} // namespace common
} // namespace byteps
