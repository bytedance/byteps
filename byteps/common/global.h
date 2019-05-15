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

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include <nccl.h>
#include <map>

#include "common.h"
#include "logging.h"
#include "communicator.h"
#include "scheduled_queue.h"
#include "ps/ps.h"

namespace byteps {
namespace common {

const int ThreadNum = QueueNum;

struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
};

typedef void (*LoopFunction)();

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
    static bool IsRootDevice();
    static int GetRoot();
    static BytePSRole GetMyRole() { return _my_role; }
    static std::shared_ptr<BytePSComm> GetComm() { return _comm; }

    static BytePSScheduledQueue* GetScheduledQueue(QueueType queueType);
    static void* CreateScheduledQueue(QueueType queueType);
    static ps::KVWorker<char>* GetPS() { return _ps; }

    static bool IsTensorInitialized(const std::string &name, size_t size);
    static ps::Key GetKeyFromName(const std::string &name);
    static BPSContext& GetContextFromName(const std::string &name);
    static uint32_t GetTensorCount();

    static std::unordered_map<int, PSKV> ps_kv_;
    static PSKV& EncodeDefaultKey(int key, size_t len);

    static uint32_t GetPartitionBound() { return _partition_bytes; }

    static cudaStream_t* GetReduceStream();
    static cudaStream_t* GetBroadcastStream();
    static cudaStream_t* GetCopyDevice2HostStream();
    static cudaStream_t* GetCopyHost2DeviceStream();

    // methods to access or modify the _ready_table
    static bool IsKeyReady(int key);
    static int AddReadyCount(int key);
    static void ClearReadyCount(int key);

    static ncclComm_t getNcclComm() { return _nccl_comm; }

private:

    static std::mutex _init_mutex;
    static volatile bool _initialized;
    static volatile bool _should_shutdown;

    static int _rank;
    static int _local_rank;
    static int _size;
    static int _local_size;
    static bool _is_root_device;
    static BytePSRole _my_role;
    static std::shared_ptr<BytePSComm> _comm;

    static volatile BytePSScheduledQueue* _queues[QueueNum];
    static std::mutex _queues_mutex[QueueNum];
    static std::thread* _threads[ThreadNum];

    static ps::KVWorker<char>* _ps;
    static std::mutex _encode_mutex;
    static std::unordered_map<std::string, BPSContext> _name_to_cxt;

    static cudaStream_t* _reduce_stream;
    static cudaStream_t* _broadcast_stream;
    static cudaStream_t* _copy_device2host_stream;
    static cudaStream_t* _copy_host2device_stream;

    static uint32_t _partition_bytes;

    // (key, ready_signal_count) pair, only valid for root device
    static std::unordered_map<int, int> _ready_table;
    // use this mutex to access/modify the _ready_table
    static std::mutex _table_mutex;

    static ncclUniqueId* _nccl_id;
    static ncclComm_t _nccl_comm;
};


} // namespace common
} // namespace byteps

#endif // BYTEPS_GLOBAL_H
