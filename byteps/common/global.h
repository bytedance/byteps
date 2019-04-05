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

#ifndef BYTEPS_GLOBAL_H
#define BYTEPS_GLOBAL_H

#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include "common.h"

namespace byteps {
namespace common {

const int QueueNum = 2;
const int ThreadNum = QueueNum;
enum QueueType { PUSH, PULL };


class BytePSScheduledQueue {

public:
    void addTask(std::shared_ptr<TensorTableEntry>);
    std::shared_ptr<TensorTableEntry> peakTask();
    std::shared_ptr<TensorTableEntry> getTask();
    int pendingSize();

private:
    // TODO: use priority queue or heap
    std::deque<std::shared_ptr<TensorTableEntry>> _sq;
    std::mutex _mutex;
};

class BytePSGlobal {

public:

    static BytePSScheduledQueue* GetScheduledQueue(QueueType queueType);
    static void SetLoopThread(QueueType queueType, std::thread* t);
    static bool StartInit();
    static Status CheckInit();
    static void FinishInit();
    static bool ShouldShutdown();
    static void Shutdown();

private:

    static std::thread* _threads[ThreadNum];
    static BytePSScheduledQueue* _queues[QueueNum];
    static std::mutex _init_mutex;
    static bool _initialized;
    static bool _should_shutdown;

};


} // namespace common
} // namespace byteps

#endif // BYTEPS_GLOBAL_H
