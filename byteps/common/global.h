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

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <string>

#define OMPI_SKIP_MPICXX
#include "mpi.h"

#include "common.h"
#include "logging.h"
#include "ps/ps.h"

namespace byteps {
namespace common {

const int QueueNum = 4;
const int ThreadNum = QueueNum;
enum QueueType { REDUCE, PUSH, PULL, BROADCAST };

typedef void (*LoopFunction)();

class BytePSScheduledQueue {

public:
    void addTask(std::shared_ptr<TensorTableEntry>);
    std::shared_ptr<TensorTableEntry> peekTask();
    std::shared_ptr<TensorTableEntry> getTask();
    uint32_t pendingSize();
    void reportFinish(std::shared_ptr<TensorTableEntry> e);
    uint64_t getFinishedNum() { return _finished; }

private:
    // TODO: use priority queue or heap
    std::deque<std::shared_ptr<TensorTableEntry>> _sq;
    std::mutex _mutex;
    std::atomic<uint64_t> _finished;
};

class BytePSGlobal {

public:

    static void Init();
    static void Start(LoopFunction* func);
    static Status CheckInit();
    static bool ShouldShutdown() { return _should_shutdown; }
    static void Shutdown();

    static int GetRank() { return _rank; }
    static int GetLocalRank() { return _local_rank; }
    static int GetSize() { return _size; }
    static int GetLocalSize() { return _local_size; }

    static BytePSScheduledQueue* GetScheduledQueue(QueueType queueType);

    static ps::KVWorker<char>* GetPS() { return _ps; }

    static bool EncodeNameToKey(const std::string &name);
    static ps::Key GetKeyFromName(const std::string &name);
    static uint32_t GetTensorCount();
    
private:

    static void _InitComm();

    static std::mutex _init_mutex;
    static volatile bool _initialized;
    static volatile bool _should_shutdown;

    static int _rank;
    static int _local_rank;
    static int _size;
    static int _local_size;
    static MPI_Comm _local_comm;
    static MPI_Comm _global_comm;

    static volatile BytePSScheduledQueue* _queues[QueueNum];
    static std::mutex _queues_mutex[QueueNum];
    static std::thread* _threads[ThreadNum];

    static ps::KVWorker<char>* _ps;
    static std::mutex _encode_mutex;
    static std::unordered_map<std::string, ps::Key> _name_to_key;
};


} // namespace common
} // namespace byteps

#endif // BYTEPS_GLOBAL_H
