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

BytePSScheduledQueue::BytePSScheduledQueue(QueueType type) {
    _qt = type;
    if (type == REDUCE) {
        _credits = 4 * BytePSGlobal::GetPartitionBound(); // 4 * partition size
    }
    else {
        _credits = 34359738368; // 32GB, basically disabling credit control
    }
    _rt = nullptr;

    switch (_qt) {
        case REDUCE:
            if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
                _rt = BytePSGlobal::GetReduceTable();
            }
            break;
        case PCIE_REDUCE:
            if (BytePSGlobal::IsCrossPcieSwitch()) {
                if (BytePSGlobal::GetCpuReducer()->isRoot()) {
                    _rt = BytePSGlobal::GetPcieReduceTable();
                }
            }
            break;
        case PUSH:
            if (BytePSGlobal::IsRootDevice()) {
                _rt = BytePSGlobal::GetPushTable();
            }
            break;
        case COPYH2D:
            if (!BytePSGlobal::IsRootDevice()) {
                _rt = BytePSGlobal::GetCopyTable();
            }
            break;
        case BROADCAST:
            if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
                _rt = BytePSGlobal::GetBroadcastTable();
            }
            break;
        default:
            break;
    }
}

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
    std::lock_guard<std::mutex> lock(_mutex);
    _sq.push_back(entry);
    // TODO: below can be optimized to O(n)
    if (_qt == REDUCE) {
        std::sort(_sq.begin(), _sq.end(),
            [](std::shared_ptr<TensorTableEntry> a, std::shared_ptr<TensorTableEntry> b) {
                if (a->priority == b->priority) {
                    return (a->key < b->key); // from the first partition to the last
                }
                return (a->priority > b->priority); // from higher priority to lower
        });
    }
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
        // JYM: shall we check whether ready_event is OK?
        // Yibo: Not for now. We assume that this task has passed COORDINATE phases
        if ((*it)->key != (uint64_t)key) {
            continue;
        }
        task = *it;
        _sq.erase(it);
        _credits -= task->len;
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
