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

#include "global.h"

namespace byteps {
namespace common {

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
    std::lock_guard<std::mutex> lock(_mutex);
    _sq.push_back(entry);
    return;
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
    std::lock_guard<std::mutex> lock(_mutex);
    auto front = _sq.front();
    _sq.pop_front();
    return front;
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::peekTask() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _sq.front();
}

uint32_t BytePSScheduledQueue::pendingSize() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _sq.size();
}

void BytePSScheduledQueue::reportFinish(std::shared_ptr<TensorTableEntry> e) {
    // TODO: return credit based on TensorTableEntry
    _finished++;
    return;
}

// Define and init global variables
int BytePSGlobal::_rank = 0;
int BytePSGlobal::_local_rank = 0;
int BytePSGlobal::_size = 1;
int BytePSGlobal::_local_size = 1;
std::thread* BytePSGlobal::_threads[QueueNum] = {NULL};
volatile BytePSScheduledQueue* BytePSGlobal::_queues[QueueNum] = {NULL};
std::mutex BytePSGlobal::_init_mutex;
volatile bool BytePSGlobal::_initialized = false;
volatile bool BytePSGlobal::_should_shutdown = false;
ps::KVWorker<char>* BytePSGlobal::_ps = NULL;
std::unordered_map<std::string, ps::Key> BytePSGlobal::_name_to_key;

BytePSScheduledQueue* BytePSGlobal::GetScheduledQueue(QueueType queueType) {
    if (!_queues[queueType]) {
        _queues[queueType] = new BytePSScheduledQueue();
    }
    return (BytePSScheduledQueue*)_queues[queueType];
}

void BytePSGlobal::Init(int rank, int local_rank, int size, int local_size) {
    std::lock_guard<std::mutex> lock(_init_mutex);
    
    // We only init once
    if (_initialized) {
        return;
    }

    _rank = rank;
    _local_rank = local_rank;
    _size = size;
    _local_size = local_size;

    // init low-level ps implementation
    _ps = new ps::KVWorker<char>(0, 0);
    ps::StartAsync(0, "byteps\0");
    if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(
            0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }

    _initialized = true;
    BPS_LOG(DEBUG) << "Inited rank=" << rank << " local_rank=" << local_rank
               << " size=" << size << " local_size=" << local_size;
    return;
}

void BytePSGlobal::Start(LoopFunction* func) {
    // Start background threads
    for (int i = 0; i < ThreadNum; i++) {
        _threads[i] = new std::thread(func[i]);
        BPS_LOG(DEBUG) << "Background thread " << i << " starts.";
    }
}


const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "BytePS has not been initialized; use bps.init().");

Status BytePSGlobal::CheckInit() {
    if (_initialized) {
        return Status::OK();
    }
    else {
        return NOT_INITIALIZED_ERROR;
    }
}

void BytePSGlobal::Shutdown() {
    _should_shutdown = true;
    for (int i = 0; i < ThreadNum; i++) {
        if (_threads[i]->joinable()) {
            _threads[i]->join();
            delete _threads[i];
        }
    }
    ps::Finalize(0, true);
    return;
}

ps::Key BytePSGlobal::GetKeyFromName(const std::string &name) {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    return _name_to_key[name];
}

bool BytePSGlobal::EncodeNameToKey(const std::string &name) {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    if (_name_to_key.find(name) == _name_to_key.end()) {
        _name_to_key[name] = _name_to_key.size();
        return true;
    }
    return false;
}

uint32_t BytePSGlobal::GetTensorCount() {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    return _name_to_key.size();
}

} // namespace common
} // namespace byteps
