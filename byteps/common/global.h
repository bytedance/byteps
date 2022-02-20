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
#include <condition_variable>

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
#include "profiler.h"
#include "nccl_manager.h"
#include "utils.h"
#include "ps/ps.h"
#include "ready_table.h"
#include "scheduled_queue.h"
#include "shared_memory.h"
#include "thread_pool.h"

#if HAVE_CUDA == 1
#include "cuda/cuda_kernels.h"
#endif

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
  static bool ShouldAbortOnTimeout() { return _should_abort_on_timeout; }
  static void Shutdown();

  static int GetRank() { return _rank; }
  static int GetLocalRank() { return _local_rank; }
  static int GetSize() { return _size; }
  static int GetLocalSize() { return _local_size; }
  static int GetWorkerID() { return _worker_id; }
  static int GetPhyNodeID() { return _phy_node_id; }
  static int GetPhyNodeNum() { return _num_phy_node; }
  static int GetNumWorker() { return _num_worker; }
  static int GetWorkerLocalRoot() { return _worker_local_root; }
  static int GetServerLocalRoot() { return _server_local_root; }
  // the visible devices. Currently each rank only uses one
  // visible device. By default it is set to local_rank.
  // It can be overriden by BYTEPS_VISIBLE_DEVICE
  static int GetVisibleDevice() { return _visible_device; }
  static bool IsRootDevice() { return _is_root_device; }
  static bool IsDistributed() { return _is_distributed_job; }
  static std::string GetUUID() { return _uuid; }
  // BytePS is launched in joint mode
  static bool IsJoint() { return _is_joint; }
  static bool IsSkipH2D() { return _skip_h2d; }
  static bool ShouldSkipInputCopy() { return _skip_input_copy; }
  // IsDirectResponse:
  // 0: receiver does not directly response. Receiver performs push response after H2D copy
  // 1: receiver directly response. Receiver performs push response inside the server handler and before H2D copy
  // 2: sender does not wait for response, and receiver does not response either.
  static int IsDirectResponse();
  static bool IsTrace() { return _is_trace; }
  static bool IsProfileAlltoall() { return _prof_all2all_latency; }
  static bool IsAlltoallUsePull() { return _is_alltoall_use_pull; }
  static bool IsCrossPcieSwitch() { return _is_cross_pcie_switch; }
  static BytePSRole GetMyRole() { return _my_role; }
  static std::shared_ptr<BytePSComm> GetBasicComm() { return _basic_comm; }
  static std::shared_ptr<BytePSSharedMemory> GetSharedMemoryObj() {
    return _shm_obj;
  }

  static BytePSScheduledQueue* GetScheduledQueue(QueueType queueType);
  static void CreateScheduledQueue(QueueType queueType);
  static ps::KVWorker<char>* GetPS(size_t index = 0) { CHECK(_ps.size()); return _ps.at(index % _ps.size()); }
  // index: the KVWorker instance index. It is used when DMLC_GROUP_SIZE is set.
  static ps::KVWorker<char>* GetOrInitPS(size_t index = 0);

  // declare a tensor with the provided name, op_type, and key, returns the declared tensor key
  // name: the operation name
  // op_type: PUSH_PULL_OP, P2P_OP, etc
  // provided_key: the tensor key for this operation. If provided_key is -1, a new key will be generated.
  // session: the session id. -1 means no session is provided
  static int32_t DeclareTensor(const std::string& name, OperationType op_type,
                               int32_t provided_key, int session);
  static int32_t DeclareP2PTensor(const std::string& name, int sender, int receiver);
  static void ReDeclareTensor();
  static bool IsResuming() { return _is_resuming; }
  static void SetResumingFlag(bool flag) {_is_resuming = flag; }

  static void RegisterCompressor(const std::string& name, 
                                 std::unordered_map<std::string, std::string>& kwargs);
  static void PinMemory(void* ptr, int numa_or_gpu_index, size_t bytes, bool gpu);
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
  static uint32_t GetAlltoallBuffBound() { return _alltoall_buff_bytes; }
  static double GetAlltoallBuffFactor() { return _alltoall_buff_factor; }
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
  static cudaStream_t* GetAllgatherCopyDevice2HostStream();
  static cudaStream_t* GetAllgatherCopyHost2DeviceStream();
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
  static ReadyTable* GetP2PPullResponseTable();
  static ReadyTable* GetP2PAckTable();
  // for non-root
  static ReadyTable* GetCopyTable() { return _copy_table; }

  static ReadyTable* GetAllgatherTable() { return _allgather_table; }
  static ReadyTable* GetAllgatherBcastTable() { return _allgather_bcast_table; }
  static ReadyTable* GetAllgatherPullRespTable();
  static ReadyTable* GetAllgatherPullAckTable();
  static ReadyTable* GetAllgatherPullWorkerLocalRootRespTable();
  static ReadyTable* GetAllgatherPullWorkerLocalRootAckTable();
  // for non-root
  static ReadyTable* GetAllgatherCopyH2DTable() { return _allgather_copy_h2d_table; }

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
  // in such a case, we use a counter to record each occurrence of the operation
  static uint64_t GetSessionId(std::string name) {
    // TODO(haibin.lin): we temporarily use these two APIs to get the end-to-end latency of
    // alltoall operations: GetSessionId and MarkDone.
    // The logic should be moved to core_loops.cc later, instead of doing it in ops.cc
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
  static int GetP2PCopyGroupSize() { return _p2p_copy_group_size; }
  static bool IsP2PAckDisabled() { return _p2p_disable_pull_ack; }

  // returns true if the system should shutdown
  // returns false if it times out with target duration
  static bool WaitForShutdown(const std::chrono::seconds&);
  static int64_t GetMonitorInterval() { return _monitor_interval; }

  // feature
  static bool IsCpuAllreduceDisabled() { return _disable_cpu_allreduce; }
  static bool IsP2PDisabled() { return _disable_p2p; }
  static bool IsGpuAllreduceDisabled() { return _disable_gpu_allreduce; }

  static bool IsGpuAllgatherDisabled() { return _disable_gpu_allgather; }
  static bool IsGDRAllgather() { return _is_gdr_allgather; }

  static bool IsGDR() { return _is_gdr_allreduce; }
  static bool IsGDRGpu2Gpu() { return _gdr_allreduce_level == common::GPU2GPU; }
  static bool IsGDRKeyInited(uint64_t key, int receiver);
  static size_t GetGDRPhase1Threshold() { return _gdr_phase1_tensor_threshold; }
  static size_t GetGDRPhase2Threshold() { return _gdr_phase2_tensor_threshold; }
  static int GetGlobalReduceRoot(uint64_t key) { return Hash_DJB2(key) % _num_phy_node; }
  static ReadyTable* GetGDRPushPullTable();
  static uint64_t Hash_DJB2(uint64_t key);
  static uint64_t Hash_Naive(uint64_t key);
  static uint64_t Hash_BuiltIn(uint64_t key);
  static uint64_t Hash_SDBM(uint64_t key);
  static uint64_t Hash_Mixed_Mode(uint64_t key);
  
  // error handling
  static bool EnableErrHandling() { return _enable_err_handling; }

 private:
  static std::mutex _init_mutex;
  static volatile bool _initialized;
  static volatile bool _should_shutdown;
  static std::condition_variable _shutdown_cv;
  static std::mutex _shutdown_mu;
  // monitor frequency, measured in seconds
  static int64_t _monitor_interval;
  static bool _should_abort_on_timeout;
  static bool _enable_err_handling;
  static int _rank;
  static int _local_rank;
  static int _size;
  static int _local_size;
  static int _worker_id;
  static int _phy_node_id;
  static int _num_phy_node;
  static size_t _ps_instance_size;
  static int _worker_local_root;
  static int _server_local_root;
  static int _num_worker;
  static int _visible_device;
  static bool _is_root_device;
  static bool _is_distributed_job;
  static bool _is_joint;
  // features
  static bool _disable_cpu_allreduce;
  static bool _disable_gpu_allreduce;
  static bool _disable_p2p;

  static bool _disable_gpu_allgather;
  static bool _is_gdr_allgather;

  // p2p
  static bool _skip_h2d;
  static bool _skip_input_copy;
  // alltoall
  static std::unordered_map<std::string, uint64_t> _alltoall_session_ids;
  static std::unordered_map<std::string, uint64_t> _alltoall_completions;
  // The alltoall operations in BytePS is asynchronous. A worker i regards its
  // alltoall call to be complete as soon as:
  // 1) all the data from rank i is sent out, and
  // 2) all the data to rank i is received.
  // This means that the completion of alltoall at rank i does not indicate the
  // completion of alltoall at rank j, since rank j might not have received all its
  // data from other ranks.
  //
  // Due to such asynchronous nature, it's possible that worker i launches the
  // alltoall operation again before worker j completes the previous round of
  // alltoall, and the data from worker i to worker j will lead to confusion.
  //
  // In order to distinguish alltoall operations, BytePS introduces a the
  // `_alltoall_session_size` variable. An alltoall operation name is
  // prefixed with a session id [0, 1, ... `_alltoall_session_size`)
  // in a round-robin manner, such that workers can use to distinguish
  // consecutive alltoall calls.
  //
  // In BytePS, we recommend setting _alltoall_session_size to 2.
  static uint32_t _alltoall_session_size;
  static std::mutex _alltoall_session_mu;

  // whether the pull-based alltoall implementation will be used
  static bool _is_alltoall_use_pull;

  // is GPU direct allreduce mode
  static bool _is_gdr_allreduce;
  static GDRLevel _gdr_allreduce_level;
  static size_t _gdr_phase1_tensor_threshold;
  static size_t _gdr_phase2_tensor_threshold;
  static std::mutex _gdr_inited_key_mu;
  static std::unordered_map<uint64_t, std::unordered_map<int, bool>> _gdr_inited_key;
  static bool _is_cross_pcie_switch;
  static BytePSRole _my_role;
  static std::shared_ptr<BytePSComm> _basic_comm;
  static std::shared_ptr<BytePSSharedMemory> _shm_obj;
  // scheduled queues
  static volatile BytePSScheduledQueue* _queues[QueueNum];
  static std::mutex _queues_mutex[QueueNum];

  static std::vector<std::thread*> _threads;
  static std::unique_ptr<std::thread> _server_thread;

  static std::mutex _context_mutex;
  static std::vector<ps::KVWorker<char>*> _ps;
  static std::mutex _encode_mutex;
  static std::unordered_map<std::string, BPSContext> _name_to_cxt;
  // the next tensor key for declaration for given operation type
  // (PUSH_PULL_OP, ALLTOALL_OP), starting from 0
  static std::unordered_map<OperationType, int32_t> next_keys_;
  // the set of used tensor keys for given operation type
  static std::unordered_map<OperationType, std::unordered_set<int32_t>> used_keys_;
  // the next tensor key for declaration for send/recv
  // schema: pair_id -> tensor_id
  static std::unordered_map<int, unsigned int> p2p_next_keys_;

  static std::vector<std::string> _declared_tensors;
  static bool _is_resuming;
  // tracing
  static std::unordered_map<std::string, int> _name2end;
  static int _output_counter;
  static int _is_trace;
  static int _start_step;
  static int _end_step;
  static std::string _trace_dir;
  static bool _prof_all2all_latency;
  static bool _p2p_disable_pull_ack;

  static int _p2p_copy_group_size;

  // cuda
#if BYTEPS_BUILDING_CUDA == 1
  static cudaStream_t* _copy_device2host_stream;
  static cudaStream_t* _copy_host2device_stream;
  static cudaStream_t* _allgather_copy_device2host_stream;
  static cudaStream_t* _allgather_copy_host2device_stream;
  static std::shared_ptr<NcclManager> _nccl_manager;
#endif

  static uint32_t _partition_bytes;
  static uint32_t _min_compress_bytes;
  // How A2A Determine bound (the push buffer allocation size) in two ways:
  // 1. If output size unknown, get bound from env var `BYTEPS_P2P_PARTITION_BYTES`
  //    save to `_alltoall_buff_bytes` and each rank will allocate the same size;
  // 2. If output size known, get factor from env var `BYTEPS_ALLTOALL_MEM_FACTOR`
  //    save to `_alltoall_buff_factor` and multiply with the 1st time's tensor size,
  //    also we set a min size at GetAlltoallBuffBound to prevent cases where the first
  //    minibatch of data is too small.
  static uint32_t _alltoall_buff_bytes;
  static double _alltoall_buff_factor;

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

  static ReadyTable* _allgather_table;
  static ReadyTable* _allgather_bcast_table;
  static ReadyTable* _allgather_copy_h2d_table;

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
};


}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_GLOBAL_H
