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

#include <algorithm>

#include "logging.h"
#include "scheduled_queue.h"
#include "global.h"

namespace byteps {
namespace common {

BytePSScheduledQueue::BytePSScheduledQueue(QueueType type, uint64_t credits){
    _qt = type;
    _credits = credits;
    _rt = nullptr;
    if (BytePSGlobal::IsRootDevice()) {
        if (_qt == REDUCE) {
            _rt = BytePSGlobal::GetReduceTable();
        }
        else if (_qt == BROADCAST) {
            _rt = BytePSGlobal::GetBroadcastTable();
        }
    }
}

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
    std::lock_guard<std::mutex> lock(_mutex);
    _sq.push_back(entry);
    // TODO: below can be optimized to O(n)
    std::sort(_sq.begin(), _sq.end(),
        [](std::shared_ptr<TensorTableEntry> a, std::shared_ptr<TensorTableEntry> b) {
            if (a->priority == b->priority) {
                return (a->key < b->key); // from the first partition to the last
            }
            return (a->priority > b->priority); // from higher priority to lower
    });
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt] << " addTask: " << entry->tensor_name
                   << " key: " << entry->key << " rank: " << BytePSGlobal::GetLocalRank();
    return;
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
    std::lock_guard<std::mutex> lock(_mutex);
    std::shared_ptr<TensorTableEntry> task;
    // TODO: below can be optimized
    // If we take task from the tail, erase() can be faster
    for (auto it = _sq.begin(); it!=_sq.end(); ++it) {
        if ((*it)->ready_event) {
            if (!(*it)->ready_event->Ready()) {
                continue;
            }
        }
        if ((*it)->len > _credits) {
            continue;
        }
        if (_rt) {
            if (!_rt->IsKeyReady((*it)->key)) {
                continue;
            }
            _rt->ClearReadyCount((*it)->key);
        }
        task = *it;
        _sq.erase(it);
        _credits -= task->len;
        BPS_LOG(TRACE) << "Queue " << LogStrings[_qt] << " getTask: " << task->tensor_name
                       << " key: " << task->key << " rank: " << BytePSGlobal::GetLocalRank();
        return task;
    }
    return nullptr;
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask(int key){
    std::lock_guard<std::mutex> lock(_mutex);
    std::shared_ptr<TensorTableEntry> task;
    for (auto it = _sq.begin(); it!=_sq.end(); ++it) {
        if ((*it)->key != key) {
            continue;
        }
        task = *it;
        _sq.erase(it);
        BPS_LOG(TRACE) << "Queue " << LogStrings[_qt] << " getTask(key): " << task->tensor_name
                       << " key: " << task->key << " rank: " << BytePSGlobal::GetLocalRank();
        return task;
    }
    return nullptr;
}

uint32_t BytePSScheduledQueue::pendingSize() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _sq.size();
}

void BytePSScheduledQueue::reportFinish(int size) {
    _credits += size;
    return;
}

} // namespace common
} // namespace byteps
