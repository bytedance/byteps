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

#include <memory>
#include <malloc.h>
#include <numa.h>
#include <unistd.h>

#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "compressor/compressor.h"
#include "../server/server.h"
#include "global.h"

namespace byteps {
namespace common {

// Define and init global variables
std::mutex BytePSGlobal::_init_mutex;
volatile bool BytePSGlobal::_initialized = false;
volatile bool BytePSGlobal::_should_shutdown = false;

int BytePSGlobal::_rank = -1;
int BytePSGlobal::_local_rank = 0;
int BytePSGlobal::_size = 1;
int BytePSGlobal::_local_size = 1;
int BytePSGlobal::_worker_id = 0;
int BytePSGlobal::_phy_node_id = 0;
int BytePSGlobal::_num_phy_node = 1;
int BytePSGlobal::_local_root = -1;
int BytePSGlobal::_server_local_root = -1;
int BytePSGlobal::_num_worker = 1;
int BytePSGlobal::_visible_device = -1;
BytePSRole BytePSGlobal::_my_role;
bool BytePSGlobal::_is_root_device;
bool BytePSGlobal::_is_distributed_job = false;
bool BytePSGlobal::_is_cross_pcie_switch = false;
bool BytePSGlobal::_is_joint = false;
// all-to-all
bool BytePSGlobal::_skip_h2d = false;
bool BytePSGlobal::_skip_d2h = false;
uint32_t BytePSGlobal::_partition_bytes = 4096000;
uint32_t BytePSGlobal::_alltoall_buff_bytes = 4096000;
uint32_t BytePSGlobal::_min_compress_bytes = (1 << 16);

int BytePSGlobal::_is_trace = 0;
int BytePSGlobal::_start_step = 10;
int BytePSGlobal::_end_step = 20;
std::string BytePSGlobal::_trace_dir;
bool BytePSGlobal::_prof_all2all_latency = false;
std::unordered_map<std::string, int> BytePSGlobal::_name2end;
int BytePSGlobal::_output_counter = 0;

int BytePSGlobal::_pagesize = 0;

std::shared_ptr<BytePSComm> BytePSGlobal::_basic_comm;
std::shared_ptr<BytePSSharedMemory> BytePSGlobal::_shm_obj;
std::unordered_map<uint64_t, PSKV> BytePSGlobal::ps_kv_;
std::unordered_map<uint64_t, int64_t> BytePSGlobal::ps_kv_max_size_;
std::vector<unsigned long> BytePSGlobal::_server_accumulated_len;
unsigned long BytePSGlobal::_total_accumulated_len = 0;
std::string BytePSGlobal::_hash_knob;

volatile BytePSScheduledQueue* BytePSGlobal::_queues[QueueNum] = {NULL};
std::mutex BytePSGlobal::_queues_mutex[QueueNum];
bool BytePSGlobal::_lockless_queue = false;
std::vector<std::thread*> BytePSGlobal::_threads;
std::unique_ptr<std::thread> BytePSGlobal::_server_thread;

std::mutex BytePSGlobal::_context_mutex;
std::vector<ps::KVWorker<char>*> BytePSGlobal::_ps;
std::mutex BytePSGlobal::_encode_mutex;
ReadyTable* BytePSGlobal::_reduce_table;
ReadyTable* BytePSGlobal::_pcie_reduce_table;
ReadyTable* BytePSGlobal::_broadcast_table;
ReadyTable* BytePSGlobal::_push_table;
ReadyTable* BytePSGlobal::_cpu_reduce_table;
ReadyTable* BytePSGlobal::_cpu_bcast_table;
ReadyTable* BytePSGlobal::_cpu_bcast_finish_table;
ReadyTable* BytePSGlobal::_copy_table;
ReadyTable* BytePSGlobal::_p2p_copy_table;
bool BytePSGlobal::_is_using_reduce = false;
std::vector<int> BytePSGlobal::_reduce_roots;
// for alltoall
uint32_t BytePSGlobal::_alltoall_session_size = 2;
std::unordered_map<std::string, uint64_t> BytePSGlobal::_alltoall_session_ids;
std::unordered_map<std::string, uint64_t> BytePSGlobal::_alltoall_completions;
std::mutex BytePSGlobal::_alltoall_session_mu;
bool BytePSGlobal::_p2p_disable_pull_ack = false;
bool BytePSGlobal::_is_alltoall_use_pull = false;

// key/name declaration
std::vector<std::string> BytePSGlobal::_declared_tensors;
bool BytePSGlobal::_is_resuming = false;
std::unordered_map<std::string, BPSContext> BytePSGlobal::_name_to_cxt;
std::unordered_map<OperationType, int32_t> BytePSGlobal::next_keys_;
std::unordered_map<OperationType, std::unordered_set<int32_t>> BytePSGlobal::used_keys_;
std::unordered_map<int, unsigned int> BytePSGlobal::p2p_next_keys_;

#if BYTEPS_BUILDING_CUDA == 1
cudaStream_t* BytePSGlobal::_copy_device2host_stream = NULL;
cudaStream_t* BytePSGlobal::_copy_host2device_stream = NULL;
std::shared_ptr<NcclManager> BytePSGlobal::_nccl_manager;
#endif
std::string BytePSGlobal::_uuid;
// reduction
std::shared_ptr<CpuReducer> BytePSGlobal::_cpu_reducer;
std::shared_ptr<GpuReducer> BytePSGlobal::_gpu_reducer;

std::shared_ptr<ThreadPool> BytePSGlobal::_thread_pool;
// hash functions
std::hash<std::string> BytePSGlobal::_built_in_hash_fn;
unsigned int BytePSGlobal::_built_in_hash_coefficient;
volatile bool BytePSGlobal::_mixed_mode = false;

uint64_t BytePSGlobal::_sample_key = std::numeric_limits<uint64_t>::max();
bool BytePSGlobal::_should_sample = false;
std::atomic_int BytePSGlobal::joined_thread_cnt;
int BytePSGlobal::_p2p_copy_group_size;
BytePSScheduledQueue* BytePSGlobal::GetScheduledQueue(QueueType queueType) {
  return (BytePSScheduledQueue*)_queues[queueType];
}

void BytePSGlobal::CreateScheduledQueue(QueueType queueType) {
  std::lock_guard<std::mutex> lock(_queues_mutex[queueType]);
  if (!_queues[queueType]) {
    _queues[queueType] = new BytePSScheduledQueue(queueType);
  }
  return;
}

void BytePSGlobal::Init() {
  std::lock_guard<std::mutex> lock(_init_mutex);

  // We only init once
  if (_initialized) {
    return;
  }

  // Set the profiling-related variables
  _is_trace =
      getenv("BYTEPS_TRACE_ON") ? atoi(getenv("BYTEPS_TRACE_ON")) : _is_trace;
  _start_step = getenv("BYTEPS_TRACE_START_STEP")
                    ? atoi(getenv("BYTEPS_TRACE_START_STEP"))
                    : _start_step;
  _end_step = getenv("BYTEPS_TRACE_END_STEP")
                  ? atoi(getenv("BYTEPS_TRACE_END_STEP"))
                  : _end_step;
  _trace_dir = getenv("BYTEPS_TRACE_DIR")
                   ? std::string(getenv("BYTEPS_TRACE_DIR"))
                   : "./trace";

  Telemetry::InitEnv();
  // Set p2p related variables
  _prof_all2all_latency = getenv("BYTEPS_PROFILE_ALL2ALL") ? atoi(getenv("BYTEPS_PROFILE_ALL2ALL")) : false;
  _uuid = getenv("BYTEPS_UUID") ? std::string(getenv("BYTEPS_UUID")) : BYTEPS_DEFAULT_UUID;
  _is_joint = std::string(getenv("DMLC_ROLE")) == "joint" ? true : false;
  _skip_h2d = getenv("BYTEPS_P2P_SKIP_H2D") ? atoi(getenv("BYTEPS_P2P_SKIP_H2D")) : false;
  _skip_d2h = getenv("BYTEPS_P2P_SKIP_D2H") ? atoi(getenv("BYTEPS_P2P_SKIP_D2H")) : false;
  _lockless_queue = getenv("BYTEPS_LOCKLESS_QUEUE") ? atoi(getenv("BYTEPS_LOCKLESS_QUEUE")) : false;
  _alltoall_session_size = getenv("BYTEPS_ALLTOALL_SESSION_SIZE") ? atoi(getenv("BYTEPS_ALLTOALL_SESSION_SIZE")) : 2;
  _p2p_copy_group_size = getenv("BYTEPS_ALLTOALL_COPY_GROUP_SIZE") ? atoi(getenv("BYTEPS_ALLTOALL_COPY_GROUP_SIZE")) : 16;
  _is_alltoall_use_pull = getenv("BYTEPS_ALL2ALL_USE_PULL") ? atoi(getenv("BYTEPS_ALL2ALL_USE_PULL")) : false;
  BPS_LOG(INFO) << "Joint=" << _is_joint << ", skip_h2d=" << _skip_h2d
                << ", skip_d2h=" << _skip_d2h << ", trace=" << _is_trace
                << ", session_size=" << _alltoall_session_size
                << ", use_pull=" << (_is_alltoall_use_pull ? "Y" : "N");

  if (getenv("BYTEPS_WORKER_LOCAL_ROOT")) {
    _local_root = atoi(getenv("BYTEPS_WORKER_LOCAL_ROOT"));
  }
  if (_local_root == -1) {
    _local_root = _local_size - 1;
  }

  if (getenv("BYTEPS_SERVER_LOCAL_ROOT")) {
    _server_local_root = atoi(getenv("BYTEPS_SERVER_LOCAL_ROOT"));
  }
  if (_server_local_root == -1) {
    _server_local_root = _local_size - 1;
  }

  _basic_comm = std::make_shared<BytePSCommSocket>();
  _basic_comm->init(&_rank, &_size, &_local_rank, &_local_size, &_worker_id,
                    &_my_role, &_num_phy_node, &_phy_node_id);

  _is_root_device = (_my_role == LOCAL_ROOT) ? true : false;

  if (getenv("BYTEPS_DISABLE_P2P_ACK")) {
    _p2p_disable_pull_ack = atoi(getenv("BYTEPS_DISABLE_P2P_ACK"));
  }

  // should round up partition bytes in order to be page aligned
  if (getenv("BYTEPS_PARTITION_BYTES")) {
    _partition_bytes = atoi(getenv("BYTEPS_PARTITION_BYTES"));
    _alltoall_buff_bytes = _partition_bytes;
  }
  // TODO(haibin.lin): rename it to BYTEPS_ALLTOALL_BUFF_BYTES
  if (getenv("BYTEPS_P2P_PARTITION_BYTES")) {
    _alltoall_buff_bytes = atoi(getenv("BYTEPS_P2P_PARTITION_BYTES"));
  }
  if (getenv("BYTEPS_MIN_COMPRESS_BYTES")) {
    _min_compress_bytes = atoi(getenv("BYTEPS_MIN_COMPRESS_BYTES"));
  }
  _pagesize = sysconf(_SC_PAGESIZE);
  BPS_CHECK_GT(_pagesize, 0);
  _partition_bytes = RoundUp(_partition_bytes, _local_size * _pagesize);
  BPS_LOG(DEBUG) << "Partition size round up to " << _partition_bytes
                 << " (bytes)";

  BPS_CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";
  _num_worker = atoi(getenv("DMLC_NUM_WORKER"));

  if (getenv("BYTEPS_FORCE_DISTRIBUTED")) {
    _is_distributed_job = atoi(getenv("BYTEPS_FORCE_DISTRIBUTED"));
  }
  _is_distributed_job = (_num_worker > 1) ? true : _is_distributed_job;

  if (_is_distributed_job) {
    BPS_CHECK(getenv("DMLC_NUM_SERVER"))
        << "error: launch distributed job, but env DMLC_NUM_SERVER not set";

    // set hash function
    std::string default_hash_knob = std::string("djb2");
    if (_is_joint) {
      default_hash_knob = std::string("djb2-colocate");
    }
    _hash_knob = std::string(
        getenv("BYTEPS_KEY_HASH_FN") ? getenv("BYTEPS_KEY_HASH_FN") :
        default_hash_knob);
    _mixed_mode = getenv("BYTEPS_ENABLE_MIXED_MODE")
                      ? atoi(getenv("BYTEPS_ENABLE_MIXED_MODE"))
                      : false;
    if (_mixed_mode) {
      _hash_knob = std::string("mixed");
    }
    BPS_LOG(DEBUG) << "Using key hash function type: " << _hash_knob;
    if (!_hash_knob.compare(std::string("built_in"))) {
      _built_in_hash_coefficient =
          getenv("BYTEPS_BUILT_IN_HASH_COEF")
              ? atoi(getenv("BYTEPS_BUILT_IN_HASH_COEF"))
              : 1;
      BPS_LOG(DEBUG) << "The built in hash coefficient is set to "
                     << _built_in_hash_coefficient;
    }

    // set server load counter
    int num_server = atoi(getenv("DMLC_NUM_SERVER"));
    for (int i = 0; i < num_server; ++i) _server_accumulated_len.push_back(0);
  }

  BPS_LOG(DEBUG) << "Number of workers=" << _num_worker << ", launching a "
                 << (IsDistributed() ? "" : "non-") << "distributed job";

  _shm_obj = std::make_shared<BytePSSharedMemory>();  // share memory obj

  // Init NCCL
#if BYTEPS_BUILDING_CUDA == 1
  auto bps_visible_device = getenv("BYTEPS_VISIBLE_DEVICE");
  auto visible_device = getenv("CUDA_VISIBLE_DEVICES");
  if (bps_visible_device) {
    _visible_device = atoi(bps_visible_device);
  } else if (visible_device) {
    auto visible_device_str = std::string(visible_device);
    std::unordered_set<int> device_set;
    size_t pos_begin = 0;
    for (size_t i = 1; i < visible_device_str.size(); i++) {
      if (visible_device_str[i] == ',') {
        size_t pos_end = i;
        auto last_device = visible_device_str.substr(pos_begin, pos_end);
        if (last_device.size()) {
          auto curr_device = atoi(last_device.c_str());
          device_set.insert(curr_device);
        }
        pos_begin = i + 1;
      }
    }
    auto last_device = visible_device_str.substr(pos_begin);
    if (last_device.size()) {
      auto curr_device = atoi(last_device.c_str());
      device_set.insert(curr_device);
    }
    int num_devices = device_set.size();
    BPS_CHECK(num_devices > 0) << num_devices;
    _visible_device = _local_rank % num_devices;
  }

  // Set to associated GPU
  CUDA_CALL(cudaSetDevice(_visible_device));

  _nccl_manager = std::make_shared<NcclManager>(_basic_comm);
  _is_cross_pcie_switch = (_local_size > _nccl_manager->GetSize());
  // Bind to NUMA node
  if (_is_cross_pcie_switch) {
    auto numa_index = (GetPcieSwitchIndex() > numa_max_node())
                          ? numa_max_node()
                          : GetPcieSwitchIndex();
    numa_bind(numa_parse_nodestring(std::to_string(numa_index).c_str()));
  }
#endif

  // Init CPU Reducer
  if (_is_cross_pcie_switch) {
    _cpu_reducer = std::make_shared<CpuReducer>(_basic_comm);
  } else if (_is_joint) {
    // cpu reducer is used for CPU allreduce and alltoall
    _cpu_reducer = std::make_shared<CpuReducer>(nullptr);
  }
  _gpu_reducer = std::make_shared<GpuReducer>();

  // ready table for send & recv
  if (_is_joint) {
    server::BytePSServer::InitP2PCopyTable();
  }
  // ReadyTable for Push & Pull
  if (_is_root_device) {
    _push_table = new ReadyTable(_local_size - 1, "PUSH");
    _cpu_reduce_table = new ReadyTable(_local_size - 1, "CPU_REDUCE");
    _cpu_bcast_finish_table = new ReadyTable(_local_size - 1, "CPU_BCAST_FINISH");
  } else {
    _copy_table = new ReadyTable(1, "COPY");
    _cpu_reduce_table = new ReadyTable(1, "CPU_REDUCE");
    _cpu_bcast_table = new ReadyTable(1, "CPU_BCAST");
  }

  if (_is_root_device) {
    size_t pool_size = 4;
    if (getenv("BYTEPS_THREADPOOL_SIZE")) {
      pool_size = atoi(getenv("BYTEPS_THREADPOOL_SIZE"));
      _thread_pool.reset(new ThreadPool(pool_size));
    }
  }

#if BYTEPS_BUILDING_CUDA == 1
  // ReadyTable for cross-PCIe-switch reduce
  if (_is_cross_pcie_switch) {
    if (_cpu_reducer->isRoot()) {
      _pcie_reduce_table =
          new ReadyTable(GetPcieSwitchNum() - 1, "PCIE_REDUCE");
    }
  }

  // ReadyTable for per-PCIe-switch NCCL calls
  if (_nccl_manager->IsSignalRoot()) {
    _reduce_table = new ReadyTable(GetPcieSwitchSize() - 1, "NCCL_REDUCE");
    _broadcast_table =
        new ReadyTable(GetPcieSwitchSize() - 1, "NCCL_BROADCAST");
  }

  // Configure the reduce strategy
  if (getenv("BYTEPS_REDUCE_ROOTS")) {
    BPS_CHECK(!_is_cross_pcie_switch)
        << "BYTEPS_REDUCE_ROOTS cannot be used with BYTEPS_PCIE_SWITCH_SIZE.";
    _is_using_reduce = true;
    auto roots_str = std::string(getenv("BYTEPS_REDUCE_ROOTS"));
    BPS_LOG(DEBUG) << "Setting roots for reduce:" << roots_str;
    std::stringstream roots_ss(roots_str);
    for (int i; roots_ss >> i;) {
      _reduce_roots.push_back(i);
      if (roots_ss.peek() == ',') {
        roots_ss.ignore();
      }
    }
  }

  // Create CUDA streams for GPU-CPU copies
  _copy_host2device_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 1);
  _copy_device2host_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 1);
  CUDA_CALL(cudaStreamCreateWithFlags(_copy_host2device_stream,
                                      cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamCreateWithFlags(_copy_device2host_stream,
                                      cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamSynchronize(*_copy_host2device_stream));
  CUDA_CALL(cudaStreamSynchronize(*_copy_device2host_stream));
#endif
  // Create queues
  for (int i = 0; i < QueueNum; i++) {
    auto type = static_cast<QueueType>(i);
    BytePSGlobal::CreateScheduledQueue(type);
  }

  joined_thread_cnt = 0;

  _initialized = true;
  BPS_LOG(DEBUG) << "Inited rank=" << _rank << " local_rank=" << _local_rank
                 << " size=" << _size << " local_size=" << _local_size
                 << " worker_id=" << _worker_id << " pid=" << getpid();

  if (getenv("BYTEPS_DEBUG_SAMPLE_TENSOR")) {
    _should_sample = true;
    _sample_key = strtoull(getenv("BYTEPS_DEBUG_SAMPLE_TENSOR"), nullptr, 0);
    BPS_LOG(DEBUG) << "_sample_key " << _sample_key;
  }
  return;
}

ps::KVWorker<char>* BytePSGlobal::GetOrInitPS(int index) {
  // we reuse _init_mutex, because BytePS should have been inited
  bool need_ps = IsDistributed() && (_my_role == BytePSRole::LOCAL_ROOT || _is_joint);
  std::lock_guard<std::mutex> lock(_init_mutex);
  // init low-level ps implementation
  if (_ps.empty() && need_ps) {
    const char* role_val = getenv("DMLC_ROLE");
    CHECK(role_val) << "DMLC_ROLE is not set";
    ps::Node::Role ps_role = ps::GetRole(std::string(role_val));
    auto val = getenv("DMLC_GROUP_SIZE");
    int group_size = val ? atoi(val) : 1;
    BPS_LOG(DEBUG) << "Initializing PS worker. rank=" << _worker_id
                    << " role=" << ps_role << " group_size=" << group_size;
    ps::StartPS(0, ps_role, _worker_id, true, "byteps\0");
    for (int i = 0; i < group_size; ++i) {
      _ps.push_back(new ps::KVWorker<char>(0, 0, i));
    }
    if (ps_role == ps::Node::JOINT) {
      _server_thread = std::unique_ptr<std::thread>(new std::thread(server::BytePSServer::Init, _worker_id));
    }
    // TODO: support resume
    BPS_LOG(DEBUG) << "PS rank " << _worker_id << " initialized. num_server="
                    << ps::NumServers() << ". num_worker=" << ps::NumWorkers()
                    << ". group_size=" << group_size;
  }
  if (_ps.size() > index) {
    return _ps.at(index);
  }
  return nullptr;
}

void BytePSGlobal::Start(const std::vector<LoopFunction>& func) {
  // Start background threads
  for (size_t i = 0; i < func.size(); i++) {
    _threads.push_back(new std::thread(func[i]));
  }
  BPS_LOG(DEBUG) << "Started " << func.size()
                 << " background threads. local_rank=" << _local_rank;
}

const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "BytePS has not been initialized; use bps.init().");

Status BytePSGlobal::CheckInit() {
  if (_initialized) {
    return Status::OK();
  } else {
    return NOT_INITIALIZED_ERROR;
  }
}

void BytePSGlobal::Shutdown() {
  BPS_LOG(DEBUG) << "Shutdown BytePS: start to clean the resources"
                 << " (rank=" << _local_rank << ")";
  _should_shutdown = true;
  int total_thread_num = _threads.size();

  BPS_LOG(DEBUG) << "Shutdown BytePS: joining " << total_thread_num << " threads"
                 << " (rank=" << _local_rank << ")";
  for (size_t i = 0; i < _threads.size(); i++) {
    if (_threads[i]->joinable()) {
      _threads[i]->join();
      delete _threads[i];
      _threads[i] = NULL;
    }
  }

  BPS_LOG(DEBUG) << "Shutdown BytePS: joined " << BytePSGlobal::joined_thread_cnt.fetch_add(0)
    << " threads" << " expecting " << total_thread_num << "threads";
  while (!IsAllThreadFinish(total_thread_num)) {
    // wait until all threads joined
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  BPS_LOG(DEBUG) << "Shutdown BytePS: joined " << total_thread_num << " threads"
                 << " (rank=" << _local_rank << ")";

  for (size_t i = 0; i < QueueNum; i++) {
    if (_queues[i]) {
      delete _queues[i];
      _queues[i] = NULL;
    }
  }
  BPS_LOG(DEBUG) << "Shutdown PS ... ";
  if (!_ps.empty()) {
    // shutdown _ps and wait for the completion acks of other workers/servers
    BPS_LOG(DEBUG) << "Shutdown BytePS: waiting for worker to finalize"
                   << " (rank=" << _local_rank << ")";
    ps::Finalize(0, ps::Node::WORKER, true);
    for (auto worker : _ps) delete worker;
    _ps.clear();
  }
  BPS_LOG(DEBUG) << "Shutdown BytePS: worker finalized"
                 << " (rank=" << _local_rank << ")";
  if (_server_thread) {
    BPS_LOG(DEBUG) << "Shutdown BytePS: waiting for server to finalize"
                   << " (rank=" << _local_rank << ")";
    _server_thread->join();
    BPS_LOG(DEBUG) << "Shutdown BytePS: server finalized"
                   << " (rank=" << _local_rank << ")";
  }

#if BYTEPS_BUILDING_CUDA == 1
  if (_copy_device2host_stream) {
    CUDA_CALL(cudaStreamDestroy(*_copy_device2host_stream));
    _copy_device2host_stream = NULL;
  }
  if (_copy_host2device_stream) {
    CUDA_CALL(cudaStreamDestroy(*_copy_host2device_stream));
    _copy_host2device_stream = NULL;
  }
#endif

  if (_reduce_table) {
    delete _reduce_table;
    _reduce_table = NULL;
  }
  if (_pcie_reduce_table) {
    delete _pcie_reduce_table;
    _pcie_reduce_table = NULL;
  }
  if (_broadcast_table) {
    delete _broadcast_table;
    _broadcast_table = NULL;
  }
  if (_push_table) {
    delete _push_table;
    _push_table = NULL;
  }

  if (_copy_table) {
    delete _copy_table;
    _copy_table = NULL;
  }

  if (_p2p_copy_table) {
    delete _p2p_copy_table;
    _p2p_copy_table = NULL;
  }

  _basic_comm.reset();
  _shm_obj.reset();
  _cpu_reducer.reset();
#if BYTEPS_BUILDING_CUDA == 1
  _nccl_manager.reset();
#endif

  // reset state, ignore profiling state
  BPS_LOG(DEBUG) << "Clear BytePS state";
  _threads.clear();
  joined_thread_cnt = 0;
  _name_to_cxt.clear();
  _server_accumulated_len.clear();
  _total_accumulated_len = 0;
  ps_kv_.clear();
  next_keys_.clear();
  _initialized = false;
  _should_shutdown = false;

  BPS_LOG(DEBUG) << "Shutdown BytePS: all BytePS resources has been cleaned"
                 << " (rank=" << _local_rank << ")";
  return;
}

BPSContext& BytePSGlobal::GetContextFromName(const std::string& name) {
  std::lock_guard<std::mutex> lock(_context_mutex);
  BPS_CHECK(_name_to_cxt.find(name) != _name_to_cxt.end())
      << name << " is not initialized";
  return _name_to_cxt[name];
}

int32_t BytePSGlobal::IsTensorDeclaredP2P(const std::string& name, int sender, int receiver) {
  std::lock_guard<std::mutex> lock(_context_mutex);
  auto _ctx = _name_to_cxt.find(name);
  if (_ctx == _name_to_cxt.end()) {
    if (std::find(_declared_tensors.begin(), _declared_tensors.end(), name) == _declared_tensors.end()) {
      _declared_tensors.push_back(name);
    }
    _name_to_cxt[name].initialized = false;
    _name_to_cxt[name].tensor_name = name.c_str();  // disable copy-on-write
    _name_to_cxt[name].op_type = P2P_OP;
    // the next key starts from 0 per send/recv pair
    int send_recv_pair = (sender << 16) + receiver;
    _name_to_cxt[name].sender = sender;
    _name_to_cxt[name].receiver = receiver;
    // TODO(haibin.lin): self send/recv is not yet implemented
    BPS_CHECK(sender != receiver);
    _name_to_cxt[name].declared_key = p2p_next_keys_[send_recv_pair]++;;
    BPS_LOG(DEBUG) << "Declared p2p tensor " << name
                   << ", declared key (not PS key): "
                   << _name_to_cxt[name].declared_key
                   << ", worker_id=" << BytePSGlobal::GetWorkerID()
                   << ", my_rank=" << BytePSGlobal::GetRank()
                   << ", sender=" << sender
                   << ", receiver=" << receiver;
  }
  return _name_to_cxt[name].declared_key;
}

int32_t BytePSGlobal::IsTensorDeclared(const std::string& name, OperationType op_type, int32_t provided_key) {
  std::lock_guard<std::mutex> lock(_context_mutex);
  if (_name_to_cxt.find(name) == _name_to_cxt.end()) {
    if (std::find(_declared_tensors.begin(), _declared_tensors.end(), name) ==
        _declared_tensors.end()) {
      _declared_tensors.push_back(name);
    }
    _name_to_cxt[name].initialized = false;
    _name_to_cxt[name].tensor_name = name.c_str();  // disable copy-on-write
    auto& used_key_set = used_keys_[op_type];
    if (provided_key == -1) {
      // generate a new key
      provided_key = next_keys_[op_type]++;
      while (used_key_set.find(provided_key) != used_key_set.end()) {
        provided_key = next_keys_[op_type]++;
      }
    } else {
      BPS_CHECK(used_key_set.find(provided_key) == used_key_set.end()) << provided_key;
    }
    _name_to_cxt[name].declared_key = (ps::Key) provided_key;
    // mark the current key as used
    used_key_set.insert(provided_key);
    _name_to_cxt[name].op_type = op_type;
    BPS_LOG(DEBUG) << "Declared tensor " << name
                   << ", declared key (not PS key): "
                   << _name_to_cxt[name].declared_key
                   << " rank=" << BytePSGlobal::GetRank();
  }
  return _name_to_cxt[name].declared_key;
}

void BytePSGlobal::ReDeclareTensor() {
  for (auto name : _declared_tensors) {
    BPS_LOG(DEBUG) << "Redeclare tensor " << name;
    BytePSGlobal::IsTensorDeclared(name, PUSH_PULL_OP, -1);
  }
}

void BytePSGlobal::RegisterCompressor(
    const std::string& name,
    std::unordered_map<std::string, std::string>& kwargs) {
  std::lock_guard<std::mutex> lock(_context_mutex);
  BPS_CHECK(_name_to_cxt.find(name) != _name_to_cxt.end())
      << name << " is not initialized";
  _name_to_cxt[name].kwargs = std::move(kwargs);
}

void BytePSGlobal::PinMemory(void* ptr, int device_id, size_t bytes) {
  auto worker = GetOrInitPS(0);
  CHECK(_ps.size() == 1);
  bool gpu = true;
  if (BytePSGlobal::IsAlltoallUsePull()) {
    ps::Postoffice::GetServer()->van()->PinMemory(ptr, bytes, gpu);
    ps::Postoffice::GetWorker()->van()->PinMemory(ptr, bytes, gpu);
  } else {
    ps::Postoffice::GetWorker()->van()->PinMemory(ptr, bytes, gpu);
  }
  BPS_LOG(INFO) << "Pinned memory " << ptr << " device_id=" << device_id << " bytes=" << bytes;
}

// Append for communication traces
void BytePSGlobal::SetProfileFlag(BytePSContext* ctxt) {
  if (_is_trace == 1) {
    // Enable trace, check the start and end step
    BPS_CHECK(_start_step >= 1 && _end_step > _start_step)
        << "BYTEPS_TRACE_START_STEP must be larger than 1, "
        << "BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP.";
    if (ctxt->step_cnt == _start_step - 1) {
      ctxt->profile_flag = true;
      BytePSGlobal::Who2beOutput(ctxt->tensor_name);
    } else if (ctxt->step_cnt == _end_step) {
      ctxt->profile_flag = false;
      if (BytePSGlobal::IsAllTensorOutput(ctxt->tensor_name)) {
        std::thread _t(BytePSGlobal::OutputTraces);
        _t.detach();
      }
    }
  } else {
    ctxt->profile_flag = false;
  }
}

void BytePSGlobal::EmitTrace(std::ostream* os, const BPSCommTime* ret,
                             BytePSContext* ctxt) {
  std::string tid = (ret->key == -1) ? "total" : std::to_string(ret->key);
  std::string para_name = "Comm." + ctxt->tensor_name;
  std::string para_name_type =
      (ret->key == -1) ? para_name : para_name + "." + LogStrings[ret->type];
  (*os) << "        {\n"
        << "            \"ph\": \"X\",\n"
        << "            \"args\": {\n"
        << "                \"name\": \"" << para_name << "\"\n"
        << "            },\n"
        << "            \"pid\": \"" << para_name << "\",\n"
        << "            \"name\": \"" << para_name_type << "\",\n"
        << "            \"ts\": " << ret->start_t << ",\n"
        << "            \"dur\": " << ret->dur << ",\n"
        << "            \"tid\": \"" << tid << "\",\n"
        << "            \"cat\": \"Comm\"\n"
        << "        }";
}

void BytePSGlobal::Who2beOutput(const std::string& name) {
  std::lock_guard<std::mutex> lock(_context_mutex);
  if (_name2end.find(name) == _name2end.end()) {
    _name2end[name] = 1;
    _output_counter += 1;
  }
}

bool BytePSGlobal::IsAllTensorOutput(const std::string& name) {
  std::lock_guard<std::mutex> lock(_context_mutex);
  BPS_CHECK(_name2end.find(name) != _name2end.end())
      << "Output tensor must been registered to recorder first";
  //  _output_counter decreases by 1 to confirm the arrival of this tensro
  _output_counter -= 1;
  if (_output_counter == 0)
    return true;
  else
    return false;
}

void BytePSGlobal::OutputTraces() {
  // Asynchronously output communication traces
  auto trace_path =
      _trace_dir + "/" + std::to_string(_rank) + "/comm.json";
  // Output these traces
  std::ofstream file;
  file.open(trace_path);
  file << "{" << std::endl;
  file << "    \"traceEvents\": [" << std::endl;
  auto first = true;
  for (std::unordered_map<std::string, int>::iterator iter = _name2end.begin();
       iter != _name2end.end(); iter++) {
    BPSContext* ctxt = &_name_to_cxt[iter->first];
    while (ctxt->comm_time.size() > 0) {
      BPSCommTime* ret = ctxt->comm_time.front();
      if (!first)
        file << ",\n";
      else
        first = false;
      BytePSGlobal::EmitTrace(&file, ret, ctxt);
      ctxt->comm_time.pop();
    }
    while (!ctxt->part_comm_time.empty()) {
      auto part_id = ctxt->part_comm_time.begin()->first;
      auto& type2part_comm_time = ctxt->part_comm_time.begin()->second;
      BPS_CHECK(!type2part_comm_time.empty())
          << "type2part_comm_time should not be empty";
      while (!type2part_comm_time.empty()) {
        auto type = type2part_comm_time.begin()->first;
        auto& _part_comm_time_queue = type2part_comm_time.begin()->second;
        BPS_CHECK(_part_comm_time_queue.size() > 0)
            << "_part_comm_time_queue should not be empty";
        while (_part_comm_time_queue.size() > 0) {
          BPSCommTime* ret = _part_comm_time_queue.front();
          if (!first)
            file << ",\n";
          else
            first = false;
          BytePSGlobal::EmitTrace(&file, ret, ctxt);
          _part_comm_time_queue.pop();
        }
        type2part_comm_time.erase(type);
      }
      // if the unordered_map becomes empty, all the traces of this part_id has
      // been read, delete this part_id
      ctxt->part_comm_time.erase(part_id);
    }
  }
  file << "\n" << std::endl;
  file << "    ]," << std::endl;
  file << "    \"displayTimeUnit\": \"ms\"" << std::endl;
  file << "}" << std::endl;
  // BPS_LOG(TRACE) << "Communication traces output done!";
  std::cout << "Local rank " << _local_rank
            << ": communication traces output done!" << std::endl;
}

uint64_t BytePSGlobal::Hash_Mixed_Mode(uint64_t key) {
  const int num_server_total =
      ps::Postoffice::Get()->GetServerKeyRanges().size();
  const int num_worker_total = GetNumWorker();
  size_t num_server_noncolocate = num_server_total - num_worker_total;
  size_t num_server_colocate = num_worker_total;

  // The bound should be larger than num_server_total
  // in order to cover each server, but it also
  // cannot be too large because it might cause unbalance
  auto bound = getenv("BYTEPS_MIXED_MODE_BOUND")
                   ? atoi(getenv("BYTEPS_MIXED_MODE_BOUND"))
                   : 101;
  BPS_CHECK_GE(bound, num_server_total);
  auto ratio =
      (2.0 * num_server_noncolocate * (num_worker_total - 1)) /
      ((num_worker_total) * (num_worker_total + num_server_noncolocate) -
       2 * num_server_noncolocate);
  BPS_CHECK_LE(ratio, 1)
      << "number of (non-colocate servers) > number of (worker)"
      << ", which is not permitted in the mixed mode";
  BPS_CHECK_GE(ratio, 0);
  auto threshold = ratio * bound;

  auto hash_res = Hash_DJB2(key) % bound;
  if (hash_res < threshold) {  // assign for non-colocate servers
    return Hash_DJB2(hash_res) % num_server_noncolocate;
  } else {  // assign for colocate servers
    return num_server_noncolocate + (Hash_DJB2(hash_res) % num_server_colocate);
  }
}

uint64_t BytePSGlobal::Hash_Naive(uint64_t key) {
  return ((key >> 16) + (key % 65536)) * 9973;
}
uint64_t BytePSGlobal::Hash_BuiltIn(uint64_t key) {
  auto str = std::to_string(key).c_str();
  return _built_in_hash_fn(str) * _built_in_hash_coefficient;
}

uint64_t BytePSGlobal::Hash_DJB2(uint64_t key) {
  auto str = std::to_string(key).c_str();
  uint64_t hash = 5381;
  int c;
  while ((c = *str)) {  // hash(i) = hash(i-1) * 33 ^ str[i]
    hash = ((hash << 5) + hash) + c;
    str++;
  }
  return hash;
}

uint64_t BytePSGlobal::Hash_SDBM(uint64_t key) {
  auto str = std::to_string(key).c_str();
  uint64_t hash = 0;
  int c;
  while ((c = *str)) {  // hash(i) = hash(i-1) * 65599 + str[i]
    hash = c + (hash << 6) + (hash << 16) - hash;
    str++;
  }
  return hash;
}

PSKV BytePSGlobal::EncodeP2PKey(uint64_t key, size_t len, int receiver) {
  auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
  const int num_servers = krs.size();
  BPS_CHECK_GT(num_servers, 0);
  // send it to the target server
  int server = receiver;
  BPS_CHECK_LT(server, num_servers)
    << "server=" << server << ", num_servers=" << num_servers;
  ps::Key ps_key = krs[server].begin() + key;
  PSKV pskv;
  // initialization
  BPS_CHECK_LT(ps_key, krs[server].end());
  pskv.keys.push_back(ps_key);
  pskv.lens.push_back(len);
  pskv.size = len;
  // ps_kv_max_size_[ps_key] = len;
  BPS_LOG(TRACE) << "key " << key << " is encoded to " << pskv.keys[0]
                 << ", assigned to server " << server;
  return pskv;
}

PSKV& BytePSGlobal::EncodeDefaultKey(uint64_t key, size_t len) {
  std::lock_guard<std::mutex> lock(_encode_mutex);
  PSKV& pskv = ps_kv_[key];
  if (!pskv.keys.empty()) {
    if (len > 0 && pskv.size != len) {
      pskv.size = len;
      pskv.lens[0] = len;
    }
  } else {
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    const int num_servers = krs.size();
    BPS_CHECK_GT(num_servers, 0);
    // send it to a single random picked server
    int server = 0;
    if (!_hash_knob.compare(std::string("naive"))) {
      server = Hash_Naive(key) % num_servers;
    } else if (!_hash_knob.compare(std::string("built_in"))) {
      server = Hash_BuiltIn(key) % num_servers;
    } else if (!_hash_knob.compare(std::string("djb2"))) {
      server = Hash_DJB2(key) % num_servers;
    } else if (!_hash_knob.compare(std::string("djb2-colocate"))) {
      server = Hash_DJB2(key) % _num_phy_node;
      server = server * _local_size + _server_local_root;
    } else if (!_hash_knob.compare(std::string("sdbm"))) {
      server = Hash_SDBM(key) % num_servers;
    } else if (!_hash_knob.compare(std::string("mixed"))) {
      BPS_CHECK(_mixed_mode)
          << "mixed mode should also set: BYTEPS_ENABLE_MIXED_MODE";
      server = Hash_Mixed_Mode(key);
      CHECK_LT(server, num_servers);
    } else {
      BPS_CHECK(0) << "Unsupported BYTEPS_KEY_HASH_FN, "
                   << "must be one of [naive, built_in, djb2, sdbm]";
    }

    _server_accumulated_len[server] += len;
    _total_accumulated_len += len;
    BPS_LOG(DEBUG) << "key " << key << " assigned to server " << server
                   << ", accumulated workload for this server is "
                   << _server_accumulated_len[server] << " ("
                   << (100.0 * _server_accumulated_len[server] /
                       _total_accumulated_len)
                   << "%)";

    ps::Key ps_key = krs[server].begin() + key;
    BPS_CHECK_LT(ps_key, krs[server].end());
    pskv.keys.push_back(ps_key);
    pskv.lens.push_back(len);
    pskv.size = len;
  }
  BPS_LOG(TRACE) << "key " << key << " is encoded to " << pskv.keys[0];
  return pskv;
}

uint32_t BytePSGlobal::GetTensorCount() {
  std::lock_guard<std::mutex> lock(_context_mutex);
  return BytePSGlobal::_name_to_cxt.size();
}

#if BYTEPS_BUILDING_CUDA == 1
cudaStream_t* BytePSGlobal::GetCopyDevice2HostStream() {
  return BytePSGlobal::_copy_device2host_stream;
}

cudaStream_t* BytePSGlobal::GetCopyHost2DeviceStream() {
  return BytePSGlobal::_copy_host2device_stream;
}
#endif

bool BytePSGlobal::IsAllThreadFinish(int total_thread_num) {
  int k = BytePSGlobal::joined_thread_cnt.fetch_add(0);
  return (k == total_thread_num);
};

ReadyTable* BytePSGlobal::GetP2PCopyTable() {
  return server::BytePSServer::GetP2PCopyTable();
}

ReadyTable* BytePSGlobal::GetP2PGroupCopyTable() {
  return server::BytePSServer::GetP2PGroupCopyTable();
}

ReadyTable* BytePSGlobal::GetP2PPullResponseTable() {
  return server::BytePSServer::GetP2PPullResponseTable();
}

ReadyTable* BytePSGlobal::GetP2PAckTable() {
  return server::BytePSServer::GetP2PAckTable();
}

int BytePSGlobal::IsDirectResponse() {
  return server::BytePSServer::IsP2PDirectResponse();
}
}  // namespace common
}  // namespace byteps
