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
#define BYTEPS_DEFAULT_UUID "0000"

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
#include "gpu_reducer.h"
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
typedef void (*IndexedLoopFn)(int);

class BytePSGlobal {
 public:
  static void Init();
  static void Start(const std::vector<LoopFunction>& func);
  static void StartMultiple(const std::vector<IndexedLoopFn>& func);

  static Status CheckInit();
  static bool ShouldShutdown() { return _should_shutdown; }
  static void Shutdown();

  static int GetRank() { return _rank; }
  static int GetLocalRank() { return _local_rank; }
  static int GetSize() { return _size; }
  static int GetLocalSize() { return _local_size; }
  static int GetWorkerID() { return _worker_id; }
  static int GetPhyNodeID() { return _phy_node_id; }
  static int GetNumWorker() { return _num_worker; }
  // number of visible devices. This is usually used to make sure
  // the device ordinal of `cudaSetDevice` does not go out of bound.
  // In the CPU-only case, it is set to local_size.
  static int GetNumDevice() { return _num_devices; }
  static bool IsRootDevice() { return _is_root_device; }
  static bool IsDistributed() { return _is_distributed_job; }
  static std::string GetUUID() { return _uuid; }
  // BytePS is launched in joint mode
  static bool IsJoint() { return _is_joint; }
  static bool IsSkipH2D() { return _skip_h2d; }
  static bool IsSkipD2H() { return _skip_d2h; }
  // IsDirectResponse:
  // 0: receiver does not directly response. Receiver performs push response after H2D copy
  // 1: receiver directly response. Receiver performs push response inside the server handler and before H2D copy
  // 2: sender does not wait for response, and receiver does not response either.
  static int IsDirectResponse();
  static bool IsTrace() { return _is_trace; }
  static bool IsProfileZPush() { return _prof_zpush_latency; }
  static bool IsCrossPcieSwitch() { return _is_cross_pcie_switch; }
  static BytePSRole GetMyRole() { return _my_role; }
  static std::shared_ptr<BytePSComm> GetBasicComm() { return _basic_comm; }
  static std::shared_ptr<BytePSSharedMemory> GetSharedMemoryObj() {
    return _shm_obj;
  }

  static BytePSScheduledQueue* GetScheduledQueue(QueueType queueType, int index=0);
  static void CreateScheduledQueue(QueueType queueType);
  static bool IsQueueLockless() { return _lockless_queue; }
  static int GetLoopParallel() { return _num_loop_parallel; }
  static ps::KVWorker<char>* GetPS(int index = 0) { CHECK(_ps.size()); return _ps.at(index % _ps.size()); }
  // index: the KVWorker instance index. It is used when DMLC_GROUP_SIZE is set.
  static ps::KVWorker<char>* GetOrInitPS(int index = 0);

  static bool IsTensorDeclared(const std::string& name);
  static bool IsTensorDeclaredP2P(const std::string& name, int sender, int receiver);
  static void ReDeclareTensor();
  static bool IsResuming() { return _is_resuming; }
  static void SetResumingFlag(bool flag) {_is_resuming = flag; }

  static void RegisterCompressor(const std::string& name, 
                                 std::unordered_map<std::string, std::string>& kwargs);
  static void PinMemory(void* ptr, int numa_node, size_t bytes);
  static ps::Key GetKeyFromName(const std::string& name);
  static BPSContext& GetContextFromName(const std::string& name);
  static uint32_t GetTensorCount();

  static std::vector<unsigned long> _server_accumulated_len;
  static unsigned long _total_accumulated_len;
  static std::unordered_map<uint64_t, PSKV> ps_kv_;
  static std::unordered_map<uint64_t, int64_t> ps_kv_max_size_;
  static PSKV& EncodeDefaultKey(uint64_t key, size_t len);
  // intentionally make a copy due to variable length
  static PSKV EncodeP2PKey(uint64_t key, size_t len, int receiver);

  static uint32_t GetPartitionBound() { return _partition_bytes; }
  static uint32_t GetP2PPartitionBound() { return _p2p_partition_bytes; }
  static uint32_t GetMinCompressBound() { return _min_compress_bytes; }

  // cuda
#if BYTEPS_BUILDING_CUDA == 1
  static int GetPcieSwitchSize() { return _nccl_manager->GetSize(); }
  static int GetPcieSwitchIndex() {
    return _local_rank / _nccl_manager->GetSize();
  }
  static int GetPcieSwitchNum() {
    return _local_size / _nccl_manager->GetSize();
  }
  static cudaStream_t* GetCopyDevice2HostStream();
  static cudaStream_t* GetCopyHost2DeviceStream();
  static std::shared_ptr<NcclManager> GetNccl() { return _nccl_manager; }
#endif

  // methods to access or modify the _ready_table
  static ReadyTable* GetReduceTable() { return _reduce_table; }
  static ReadyTable* GetPcieReduceTable() { return _pcie_reduce_table; }
  static ReadyTable* GetBroadcastTable() { return _broadcast_table; }
  static ReadyTable* GetPushTable() { return _push_table; }
  static ReadyTable* GetCpuReduceTable() { return _cpu_reduce_table; }
  static ReadyTable* GetCpuBcastTable() { return _cpu_bcast_table; }
  static ReadyTable* GetCpuBcastFinishTable() { return _cpu_bcast_finish_table; }

  // reduce strategies
  static bool IsUsingReduce() { return _is_using_reduce; }
  static int GetReduceRootByKey(ps::Key k) {
    return _reduce_roots[Hash_DJB2(k) % _reduce_roots.size()];
  }

  // p2p ready tables
  static ReadyTable* GetP2PCopyTable();
  static ReadyTable* GetP2PGroupCopyTable();
  // for non-root
  static ReadyTable* GetCopyTable() { return _copy_table; }

  static std::shared_ptr<CpuReducer> GetCpuReducer() { return _cpu_reducer; }
  static std::shared_ptr<GpuReducer> GetGpuReducer() { return _gpu_reducer; }

  static bool IsTensorSampled(uint64_t key) { return _should_sample && (key == _sample_key); }

  static void SetProfileFlag(BPSContext* ctxt);
  static void EmitTrace(std::ostream* os, const BPSCommTime* ret,
                        BPSContext* ctxt);
  static void OutputTraces();
  static bool IsAllTensorOutput(const std::string& name);
  static void Who2beOutput(const std::string& name);

  static void ReportThreadFinish() { joined_thread_cnt.fetch_add(1); }
  static bool IsAllThreadFinish(int total_thread_num);
  static std::atomic_int joined_thread_cnt;
  static size_t RoundUpToPageSize(size_t x) { return RoundUp(x, _pagesize); }

  static std::shared_ptr<ThreadPool>& GetThreadPool() { return _thread_pool; }

  // in some rare cases, the application may run multiple sessions in the same process,
  // with the same name used for alltoall. this may cause name conflict.
  // in such a case, we use a counter to record each occurrance of the operation
  static uint64_t GetSessionId(std::string name) {
    _alltoall_session_mu.lock();
    uint64_t ret = _alltoall_session_ids[name]++;
    _alltoall_session_mu.unlock();
    return ret;
  }

  // used to track how many alltoall operations are done. Useful for
  // debugging hanging issues
  static void MarkDone(std::string name) {
    _alltoall_session_mu.lock();
    _alltoall_completions[name]++;
    _alltoall_session_mu.unlock();
  }

  static uint32_t GetSessionSize() { return _alltoall_session_size; }

 private:
  static std::mutex _init_mutex;
  static volatile bool _initialized;
  static volatile bool _should_shutdown;

  static int _rank;
  static int _local_rank;
  static int _size;
  static int _local_size;
  static int _worker_id;
  static int _phy_node_id;
  static int _num_phy_node;
  static int _local_root;
  static int _server_local_root;
  static int _num_worker;
  static int _num_devices;
  static bool _is_root_device;
  static bool _is_distributed_job;
  static bool _is_joint;
  // p2p
  static bool _skip_h2d;
  static bool _skip_d2h;
  // alltoall
  static std::unordered_map<std::string, uint64_t> _alltoall_session_ids;
  static std::unordered_map<std::string, uint64_t> _alltoall_completions;
  static uint32_t _alltoall_session_size;
  static std::mutex _alltoall_session_mu;

  static bool _is_cross_pcie_switch;
  static BytePSRole _my_role;
  static std::shared_ptr<BytePSComm> _basic_comm;
  static std::shared_ptr<BytePSSharedMemory> _shm_obj;
  // scheduled queues
  static volatile BytePSScheduledQueue* _queues[QueueNum];
  static std::mutex _queues_mutex[QueueNum];
  static std::vector<BytePSScheduledQueue*> _send_queues;
  static std::vector<BytePSScheduledQueue*> _p2p_d2h_queues;
  static int _num_loop_parallel;

  static std::vector<std::thread*> _threads;
  static std::unique_ptr<std::thread> _server_thread;

  static std::mutex _context_mutex;
  static std::vector<ps::KVWorker<char>*> _ps;
  static bool _lockless_queue;
  static std::mutex _encode_mutex;
  static std::unordered_map<std::string, BPSContext> _name_to_cxt;
  static std::vector<std::string> _declared_tensors;
  static bool _is_resuming;
  // tracing
  static std::unordered_map<std::string, int> _name2end;
  static int _output_counter;
  static int _is_trace;
  static int _start_step;
  static int _end_step;
  static std::string _trace_dir;
  static bool _prof_zpush_latency;

  // cuda
#if BYTEPS_BUILDING_CUDA == 1
  static cudaStream_t* _copy_device2host_stream;
  static cudaStream_t* _copy_host2device_stream;
  static std::shared_ptr<NcclManager> _nccl_manager;
#endif

  static uint32_t _partition_bytes;
  static uint32_t _p2p_partition_bytes;
  static uint32_t _min_compress_bytes;

  // (key, ready_signal_count) pair, only valid for root device
  static ReadyTable* _reduce_table;
  static ReadyTable* _pcie_reduce_table;
  static ReadyTable* _broadcast_table;
  static ReadyTable* _push_table;
  static ReadyTable* _cpu_reduce_table;
  static ReadyTable* _cpu_bcast_table;
  static ReadyTable* _cpu_bcast_finish_table;

  static ReadyTable* _p2p_copy_table;
  // (key, ready_signal_count) pair, only valid for non-root device
  static ReadyTable* _copy_table;

  static std::shared_ptr<ThreadPool> _thread_pool;

  // for reduce strategies
  static bool _is_using_reduce;
  static std::vector<int> _reduce_roots;

  static std::shared_ptr<CpuReducer> _cpu_reducer;
  static std::shared_ptr<GpuReducer> _gpu_reducer;

  // for debug sampling
  static uint64_t _sample_key;
  static bool _should_sample;

  static int AlignTo(int input, int alignment) {
    return input / alignment * alignment;
  }
  // memory related
  static int _pagesize;
  // unique identifier for the current application to avoid resource conflict
  // (e.g. shared memory name, socket name, etc)
  static std::string _uuid;
  static size_t DivUp(size_t x, size_t y) { return (x + y - 1) / y; }
  static size_t RoundUp(size_t x, size_t y) { return DivUp(x, y) * y; }

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

struct SpeedEntry {
  std::size_t ts;
  float speed;
};

class PushPullSpeed {
 public:
  static void RecordSpeed(std::shared_ptr<TensorTableEntry> task);
  static std::shared_ptr<SpeedEntry> GetSpeed();
  static bool ShouldRecord();

 private:
  static std::mutex _mtx;
  static std::queue<std::shared_ptr<SpeedEntry>> _data_points;
  static std::size_t _acc_size;
  static std::size_t _limit;
  static std::chrono::time_point<std::chrono::system_clock> _last_ts;
  static bool _initialized;
  static bool _should_record;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_GLOBAL_H
