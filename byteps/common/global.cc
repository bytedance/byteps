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

void BytePSScheduledQueue::addTask(std::share_ptr<TensorTableEntry> entry) {
    _sq.push_back(entry);
    return;
}

std::share_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
    auto front = _sq.front();
    _sq.pop_front();
    return front;
}

std::share_ptr<TensorTableEntry> BytePSScheduledQueue::peakTask() {
    return _sq.front();
}

static BytePSScheduledQueue* BytePSGlobal::GetScheduledQueue(BytePSOp op) {
    switch (op) {
        case PUSH:
            if (_pushq == NULL) {
                _pushq = new BytePSScheduledQueue();
            }
            return _pushq;
        
        case PULL:
            if (_pullq == NULL) {
                _pullq = new BytePSScheduledQueue();
            }
            return _pullq;

        default:
            break;
    }
    return NULL;
}


} // namespace common
} // namespace byteps
