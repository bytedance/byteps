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
#include "logging.h"
#include "ps/ps.h"

namespace byteps {
namespace common {

const int QueueNum = 2;
const int ThreadNum = QueueNum;
enum QueueType { PUSH, PULL };

typedef void (*LoopFunction)();

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

    static void Init(int rank, int local_rank, int size, int local_size, LoopFunction* func);
    static Status CheckInit();

    static int GetRank() {return _rank;}
    static int GetLocalRank() {return _local_rank;}
    static int GetSize() {return _size;}
    static int GetLocalSize() {return _local_size;}

    static BytePSScheduledQueue* GetScheduledQueue(QueueType queueType);

    static bool ShouldShutdown() {return _should_shutdown;}
    static void Shutdown();

private:

    static int _rank;
    static int _local_rank;
    static int _size;
    static int _local_size;
    static std::thread* _threads[ThreadNum];
    static BytePSScheduledQueue* _queues[QueueNum];
    static std::mutex _init_mutex;
    static bool _initialized;
    static bool _should_shutdown;

    static ps::KVWorker<char>* _ps;

};


} // namespace common
} // namespace byteps

#endif // BYTEPS_GLOBAL_H
