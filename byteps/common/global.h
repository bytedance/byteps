// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include <unistd.h>

#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "communicator.h"
#include "cpu_reducer.h"
#include "logging.h"
#include "nccl_manager.h"
#include "ps/ps.h"
#include "ready_table.h"
#include "scheduled_queue.h"
#include "shared_memory.h"
#include "thread_pool.h"

namespace byteps {
namespace common {

struct PSKV {
  ps::SArray<ps::Key> keys;  // n keys
  ps::SArray<int> lens;      // the length of the i-th value
  int size;
};

typedef void (*LoopFunction)();

class BytePSGlobal {
 public:
  static void Init();
  static void Start(const std::vector<LoopFunction>& func);
  static Status CheckInit();
  static bool ShouldShutdown() { return _should_shutdown; }
  static void Shutdown();

  static int GetRank() { return _rank; }
  static int GetLocalRank() { return _local_rank; }
  static int GetSize() { return _size; }
  static int GetLocalSize() { return _local_size; }
  static int GetWorkerID() { return _worker_id; }
  static int GetNumWorker() { return _num_worker; }
  static int GetPcieSwitchSize() { return _nccl_manager->GetSize(); }
  static int GetPcieSwitchIndex() {
    return _local_rank / _nccl_manager->GetSize();
  }
  static int GetPcieSwitchNum() {
    return _local_size / _nccl_manager->GetSize();
  }
  static bool IsRootDevice() { return _is_root_device; }
  static bool IsDistributed() { return _is_distributed_job; }
  static bool IsCrossPcieSwitch() { return _is_cross_pcie_switch; }
  static BytePSRole GetMyRole() { return _my_role; }
  static std::shared_ptr<BytePSComm> GetBasicComm() { return _basic_comm; }
  static std::shared_ptr<BytePSSharedMemory> GetSharedMemoryObj() {
    return _shm_obj;
  }

  static BytePSScheduledQueue* GetScheduledQueue(QueueType queueType);
  static void CreateScheduledQueue(QueueType queueType);
  static ps::KVWorker<char>* GetPS() { return _ps; }
  static ps::KVWorker<char>* GetOrInitPS();

  static bool IsTensorDeclared(const std::string& name);
  static void ReDeclareTensor();
  static bool IsResuming() { return _is_resuming; }
  static void SetResumingFlag(bool flag) {_is_resuming = flag; }

  static void RegisterCompressor(const std::string& name, 
                                 std::unordered_map<std::string, std::string>& kwargs);
  static ps::Key GetKeyFromName(const std::string& name);
  static BPSContext& GetContextFromName(const std::string& name);
  static uint32_t GetTensorCount();

  static std::vector<unsigned long> _server_accumulated_len;
  static unsigned long _total_accumulated_len;
  static std::unordered_map<uint64_t, PSKV> ps_kv_;
  static PSKV& EncodeDefaultKey(uint64_t key, size_t len);

  static uint32_t GetPartitionBound() { return _partition_bytes; }
  static uint32_t GetMinCompressBound() { return _min_compress_bytes; }

  static cudaStream_t* GetCopyDevice2HostStream();
  static cudaStream_t* GetCopyHost2DeviceStream();

  // methods to access or modify the _ready_table
  static ReadyTable* GetReduceTable() { return _reduce_table; }
  static ReadyTable* GetPcieReduceTable() { return _pcie_reduce_table; }
  static ReadyTable* GetBroadcastTable() { return _broadcast_table; }
  static ReadyTable* GetPushTable() { return _push_table; }

  // reduce strategies
  static bool IsUsingReduce() { return _is_using_reduce; }
  static int GetReduceRootByKey(ps::Key k) {
    return _reduce_roots[Hash_DJB2(k) % _reduce_roots.size()];
  }

  // for non-root
  static ReadyTable* GetCopyTable() { return _copy_table; }

  static std::shared_ptr<NcclManager> GetNccl() { return _nccl_manager; }
  static std::shared_ptr<CpuReducer> GetCpuReducer() { return _cpu_reducer; }

  static bool IsTensorSampled(uint64_t key) { return (key == _sample_key); }

  static void SetProfileFlag(BPSContext* ctxt);
  static void EmitTrace(std::ostream* os, const BPSCommTime* ret,
                        BPSContext* ctxt);
  static void OutputTraces();
  static bool IsAllTensorOutput(const std::string& name);
  static void Who2beOutput(const std::string& name);

  static void ReportThreadFinish() { joined_thread_cnt.fetch_add(1); }
  static bool IsAllThreadFinish(int total_thread_num);
  static std::atomic_int joined_thread_cnt;
  static int RoundUpToPageSize(int x) { return RoundUp(x, _pagesize); }

  static std::shared_ptr<ThreadPool>& GetThreadPool() { return _thread_pool; }

 private:
  static std::mutex _init_mutex;
  static volatile bool _initialized;
  static volatile bool _should_shutdown;

  static int _rank;
  static int _local_rank;
  static int _size;
  static int _local_size;
  static int _worker_id;
  static int _num_worker;
  static bool _is_root_device;
  static bool _is_distributed_job;
  static bool _is_cross_pcie_switch;
  static BytePSRole _my_role;
  static std::shared_ptr<BytePSComm> _basic_comm;
  static std::shared_ptr<BytePSSharedMemory> _shm_obj;

  static volatile BytePSScheduledQueue* _queues[QueueNum];
  static std::mutex _queues_mutex[QueueNum];
  static std::vector<std::thread*> _threads;

  static std::mutex _context_mutex;

  static ps::KVWorker<char>* _ps;
  static std::mutex _encode_mutex;
  static std::unordered_map<std::string, BPSContext> _name_to_cxt;
  static std::vector<std::string> _declared_tensors;
  static bool _is_resuming;

  static std::unordered_map<std::string, int> _name2end;
  static int _output_counter;
  static int _is_trace;
  static int _start_step;
  static int _end_step;
  static std::string _trace_dir;

  static cudaStream_t* _copy_device2host_stream;
  static cudaStream_t* _copy_host2device_stream;

  static uint32_t _partition_bytes;
  static uint32_t _min_compress_bytes;

  // (key, ready_signal_count) pair, only valid for root device
  static ReadyTable* _reduce_table;
  static ReadyTable* _pcie_reduce_table;
  static ReadyTable* _broadcast_table;
  static ReadyTable* _push_table;

  // (key, ready_signal_count) pair, only valid for non-root device
  static ReadyTable* _copy_table;

  static std::shared_ptr<ThreadPool> _thread_pool;

  // for reduce strategies
  static bool _is_using_reduce;
  static std::vector<int> _reduce_roots;

  static std::shared_ptr<NcclManager> _nccl_manager;
  static std::shared_ptr<CpuReducer> _cpu_reducer;

  // for debug sampling
  static uint64_t _sample_key;

  static int AlignTo(int input, int alignment) {
    return input / alignment * alignment;
  }

  static int _pagesize;
  static int DivUp(int x, int y) { return (x + y - 1) / y; }
  static int RoundUp(int x, int y) { return DivUp(x, y) * y; }

  // hash functions
  static std::string _hash_knob;
  static std::hash<std::string> _built_in_hash_fn;
  static unsigned int _built_in_hash_coefficient;
  static volatile bool _mixed_mode;
  static uint64_t Hash_Naive(uint64_t key);
  static uint64_t Hash_BuiltIn(uint64_t key);
  static uint64_t Hash_DJB2(uint64_t key);
  static uint64_t Hash_SDBM(uint64_t key);
  static uint64_t Hash_Mixed_Mode(uint64_t key);
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_GLOBAL_H
