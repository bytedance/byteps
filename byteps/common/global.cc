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

#include "global.h"
#include <malloc.h>
#include <unistd.h>
#include <numa.h>

namespace byteps {
namespace common {

// Define and init global variables
std::mutex BytePSGlobal::_init_mutex;
volatile bool BytePSGlobal::_initialized = false;
volatile bool BytePSGlobal::_should_shutdown = false;

int BytePSGlobal::_rank = 0;
int BytePSGlobal::_local_rank = 0;
int BytePSGlobal::_size = 1;
int BytePSGlobal::_local_size = 1;
int BytePSGlobal::_worker_id = 0;
int BytePSGlobal::_num_worker = 1;
BytePSRole BytePSGlobal::_my_role;
bool BytePSGlobal::_is_root_device;
bool BytePSGlobal::_is_distributed_job;
bool BytePSGlobal::_is_cross_pcie_switch;
uint32_t BytePSGlobal::_partition_bytes = 4096000;

std::shared_ptr<BytePSComm> BytePSGlobal::_basic_comm;
std::shared_ptr<BytePSSharedMemory> BytePSGlobal::_shm_obj;
std::unordered_map<uint64_t, PSKV> BytePSGlobal::ps_kv_;

volatile BytePSScheduledQueue* BytePSGlobal::_queues[QueueNum] = {NULL};
std::mutex BytePSGlobal::_queues_mutex[QueueNum];
std::vector<std::thread*> BytePSGlobal::_threads;

std::mutex BytePSGlobal::_context_mutex;
ps::KVWorker<char>* BytePSGlobal::_ps = NULL;
std::mutex BytePSGlobal::_encode_mutex;
ReadyTable* BytePSGlobal::_reduce_table;
ReadyTable* BytePSGlobal::_pcie_reduce_table;
ReadyTable* BytePSGlobal::_broadcast_table;
ReadyTable* BytePSGlobal::_push_table;

ReadyTable* BytePSGlobal::_copy_table;
std::unordered_map<std::string, BPSContext> BytePSGlobal::_name_to_cxt;
unsigned int next_key_ = 0;
cudaStream_t* BytePSGlobal::_copy_device2host_stream;
cudaStream_t* BytePSGlobal::_copy_host2device_stream;
std::shared_ptr<NcclManager> BytePSGlobal::_nccl_manager;
std::shared_ptr<CpuReducer> BytePSGlobal::_cpu_reducer;

uint64_t BytePSGlobal::_sample_key = std::numeric_limits<uint64_t>::max();

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

    _basic_comm = std::make_shared<BytePSCommSocket>();

    _basic_comm->init(&_rank, &_size, &_local_rank, &_local_size, &_worker_id, &_my_role);

    _is_root_device = (_my_role == LOCAL_ROOT) ? true : false;
    if (getenv("BYTEPS_PARTITION_BYTES")) {
        _partition_bytes = atoi(getenv("BYTEPS_PARTITION_BYTES"));
    }
    BPS_LOG(DEBUG) << "Partition bound set to " << _partition_bytes << " bytes"
                   << ", aligned to " << AlignTo(_partition_bytes, (8 * _local_size)) << " bytes";
    // alignment for Reduce-Scatter/All-Gather
    _partition_bytes = AlignTo(_partition_bytes, (8 * _local_size));

    BPS_CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";

    _num_worker = atoi(getenv("DMLC_NUM_WORKER"));

    if (getenv("BYTEPS_FORCE_DISTRIBUTED")) {
        _is_distributed_job = atoi(getenv("BYTEPS_FORCE_DISTRIBUTED"));
    }
    _is_distributed_job = (_num_worker>1) ? true : _is_distributed_job;

    if (_is_distributed_job) {
        BPS_CHECK(getenv("DMLC_NUM_SERVER")) << "error: launch distributed job, but env DMLC_NUM_SERVER not set";
    }

    BPS_LOG(DEBUG) << "Number of worker=" << _num_worker << ", launching "
                   << (IsDistributed() ? "" : "non-") << "distributed job";

    _shm_obj = std::make_shared<BytePSSharedMemory>(); // share memory obj

    if (IsDistributed() && _my_role == BytePSRole::LOCAL_ROOT) { // only the root need to do networking
        // init low-level ps implementation
        _ps = new ps::KVWorker<char>(0, 0);
        ps::StartAsync(0, "byteps\0");
        if (!ps::Postoffice::Get()->is_recovery()) {
            ps::Postoffice::Get()->Barrier(
                0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
        }
    }

    // Set to associated GPU
    CUDA_CALL(cudaSetDevice(_local_rank));

    // Init NCCL
    _nccl_manager = std::make_shared<NcclManager>(_basic_comm);
    _is_cross_pcie_switch = (_local_size > _nccl_manager->GetSize());

    // Bind to NUMA node
    if (_is_cross_pcie_switch) {
        auto numa_index = (GetPcieSwitchIndex() >  numa_max_node()) ?
                          numa_max_node() : GetPcieSwitchIndex();
        numa_bind(numa_parse_nodestring(std::to_string(numa_index).c_str()));
    }

    // Init CPU Reducer
    if (_is_cross_pcie_switch) {
        _cpu_reducer = std::make_shared<CpuReducer>(_basic_comm);
    }

    // ReadyTable for Push & Pull
    if (_is_root_device) {
        _push_table = new ReadyTable(_local_size-1, "PUSH");
    }
    else {
        _copy_table = new ReadyTable(1, "COPY");
    }

    // ReadyTable for cross-PCIe-switch reduce
    if (_is_cross_pcie_switch) {
        if (_cpu_reducer->isRoot()) {
            _pcie_reduce_table = new ReadyTable(GetPcieSwitchNum()-1, "PCIE_REDUCE");
        }
    }

    // ReadyTable for per-PCIe-switch NCCL calls
    if (_nccl_manager->IsSignalRoot()) {
        _reduce_table = new ReadyTable(GetPcieSwitchSize()-1, "NCCL_REDUCE");
        _broadcast_table = new ReadyTable(GetPcieSwitchSize()-1, "NCCL_BROADCAST");
    }

    // Create CUDA streams for GPU-CPU copies
    _copy_host2device_stream  = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    _copy_device2host_stream  = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    CUDA_CALL(cudaStreamCreateWithFlags(_copy_host2device_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_copy_device2host_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamSynchronize(*_copy_host2device_stream));
    CUDA_CALL(cudaStreamSynchronize(*_copy_device2host_stream));

    // Create queues
    for (int i = 0; i < QueueNum; i++) {
        BPS_LOG(DEBUG) << "Create schedule queue " << i;
        auto type = static_cast<QueueType>(i);
        BytePSGlobal::CreateScheduledQueue(type);
    }

    _initialized = true;
    BPS_LOG(DEBUG) << "Inited rank=" << _rank
                   << " local_rank=" << _local_rank
                   << " size=" << _size
                   << " local_size=" << _local_size
                   << " worker_id=" << _worker_id;

    if (getenv("BYTEPS_DEBUG_SAMPLE_TENSOR")) {
        _sample_key = strtoull(getenv("BYTEPS_DEBUG_SAMPLE_TENSOR"), nullptr, 0);
    }
    return;
}

void BytePSGlobal::Start(const std::vector<LoopFunction> &func) {
    // Start background threads
    for (size_t i = 0; i < func.size(); i++) {
        _threads.push_back(new std::thread(func[i]));
    }
    BPS_LOG(DEBUG) << "Started " << func.size() << " background threads. rank=" << _local_rank;
}


const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "BytePS has not been initialized; use bps.init().");

Status BytePSGlobal::CheckInit() {
    if (_initialized) {
        return Status::OK();
    }
    else {
        return NOT_INITIALIZED_ERROR;
    }
}

void BytePSGlobal::Shutdown() {
    _should_shutdown = true;
    for (size_t i = 0; i < _threads.size(); i++) {
        if (_threads[i]->joinable()) {
            _threads[i]->join();
            delete _threads[i];
        }
    }

    for (size_t i = 0; i < QueueNum; i++) {
        if (_queues[i]) {
            delete _queues[i];
        }
    }

    if (_ps) {
        ps::Finalize(0, false);
        delete _ps;
    }

    CUDA_CALL(cudaStreamDestroy(*_copy_device2host_stream));
    CUDA_CALL(cudaStreamDestroy(*_copy_host2device_stream));

    if (_reduce_table) {
        delete _reduce_table;
    }
    if (_pcie_reduce_table) {
        delete _pcie_reduce_table;
    }
    if (_broadcast_table) {
        delete _broadcast_table;
    }
    if (_push_table) {
        delete _push_table;
    }

    if (_copy_table) {
        delete _copy_table;
    }

    _basic_comm.reset();
    _shm_obj.reset();
    _cpu_reducer.reset();
    _nccl_manager.reset();

    BPS_LOG(DEBUG) << "Clear all BytePS resources";
    return;
}

BPSContext& BytePSGlobal::GetContextFromName(const std::string &name) {
    std::lock_guard<std::mutex> lock(_context_mutex);
    BPS_CHECK(_name_to_cxt.find(name) != _name_to_cxt.end()) << name << " is not initialized";
    return _name_to_cxt[name];
}

bool BytePSGlobal::IsTensorDeclared(const std::string &name) {
    std::lock_guard<std::mutex> lock(_context_mutex);
    if (_name_to_cxt.find(name) == _name_to_cxt.end()) {
        _name_to_cxt[name].initialized = false;
        _name_to_cxt[name].tensor_name = name.c_str(); // disable copy-on-write
        _name_to_cxt[name].declared_key = (ps::Key) next_key_++;
        BPS_LOG(DEBUG) << "Declared tensor " << name
                       << ", declared key (not PS key): " << _name_to_cxt[name].declared_key
                       << " rank=" << BytePSGlobal::GetLocalRank();
        return false;
    }
    return true;
}

PSKV& BytePSGlobal::EncodeDefaultKey(uint64_t key, size_t len) {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    PSKV& pskv = ps_kv_[key];
    if (!pskv.keys.empty()) {
        BPS_CHECK_EQ(static_cast<size_t>(pskv.size), len)
            << "The value size cannot be changed " << len
            << ". Key is " << key;
    } else {
        auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
        const int num_servers = krs.size();
        BPS_CHECK_GT(num_servers, 0);
        // send it to a single random picked server
        int server = (((key >> 16) + key) * 9973) % num_servers;
        BPS_LOG(DEBUG) << "key " << key << " assigned to server " << server;
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

cudaStream_t* BytePSGlobal::GetCopyDevice2HostStream() {
    return BytePSGlobal::_copy_device2host_stream;
}

cudaStream_t* BytePSGlobal::GetCopyHost2DeviceStream() {
    return BytePSGlobal::_copy_host2device_stream;
}


} // namespace common
} // namespace byteps
