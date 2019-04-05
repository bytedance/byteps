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

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::peakTask() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _sq.front();
}

int BytePSScheduledQueue::pendingSize() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _sq.size();
}

// Define and init global variables
std::thread* BytePSGlobal::_threads[QueueNum] = {NULL};
BytePSScheduledQueue* BytePSGlobal::_queues[QueueNum] = {NULL};
std::mutex BytePSGlobal::_init_mutex;
bool BytePSGlobal::_initialized = false;
bool BytePSGlobal::_should_shutdown = false;

BytePSScheduledQueue* BytePSGlobal::GetScheduledQueue(QueueType queueType) {
    if (!_queues[queueType]) {
        _queues[queueType] = new BytePSScheduledQueue();
    }
    return _queues[queueType];
}

void BytePSGlobal::SetLoopThread(QueueType queueType, std::thread* t) {
    _threads[queueType] = t;
    return;
}

// Try to start the init process
// If already inited, will return false
// Otherwise acquire the lock, return true
bool BytePSGlobal::StartInit() {
    _init_mutex.lock();
    if (_initialized) {
        _init_mutex.unlock();
        return false;
    }
    // mutex must be unlocked in FinishInit()
    return true;
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

void BytePSGlobal::FinishInit() {
    _initialized = true;
    _init_mutex.unlock();
    return;
}

bool BytePSGlobal::ShouldShutdown() {
    return _should_shutdown;
}

void BytePSGlobal::Shutdown() {
    _should_shutdown = true;
    for (int i=0; i<ThreadNum; i++) {
        if (_threads[i]->joinable()) {
            _threads[i]->join();
            delete _threads[i];
        }
    }
    return;
}


} // namespace common
} // namespace byteps
