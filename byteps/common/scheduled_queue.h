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

#ifndef BYTEPS_SCHEDULED_QUEUE_H
#define BYTEPS_SCHEDULED_QUEUE_H

#include <atomic>
#include <vector>
#include <memory>

#include "common.h"

namespace byteps {
namespace common {

class BytePSScheduledQueue {

public:
    BytePSScheduledQueue(QueueType type, uint64_t credits) {
        _qt = type;
        _credits = credits;
    }
    QueueType getQueueType() { return _qt; }
    void addTask(std::shared_ptr<TensorTableEntry>);
    std::shared_ptr<TensorTableEntry> getTask();
    uint32_t pendingSize();
    void reportFinish(int size);

private:
    // TODO: use priority queue or heap
    std::vector<std::shared_ptr<TensorTableEntry>> _sq;
    std::mutex _mutex;
    std::atomic<uint64_t> _credits;
    QueueType _qt;
};


} // namespace common
} // namespace byteps

#endif // BYTEPS_SCHEDULED_QUEUE_H