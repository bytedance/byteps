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
#include <malloc.h>
#include <unistd.h>

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
BytePSRole BytePSGlobal::_my_role;
bool BytePSGlobal::_is_root_device;
uint32_t BytePSGlobal::_partition_bytes = 1024000;
std::shared_ptr<BytePSComm> BytePSGlobal::_comm;
std::unordered_map<int, PSKV> BytePSGlobal::ps_kv_;

volatile BytePSScheduledQueue* BytePSGlobal::_queues[QueueNum] = {NULL};
std::mutex BytePSGlobal::_queues_mutex[QueueNum];
std::thread* BytePSGlobal::_threads[QueueNum] = {NULL};

ps::KVWorker<char>* BytePSGlobal::_ps = NULL;
std::mutex BytePSGlobal::_encode_mutex;
std::mutex BytePSGlobal::_table_mutex;
std::unordered_map<int, int> BytePSGlobal::_ready_table;
std::unordered_map<std::string, BPSContext> BytePSGlobal::_name_to_cxt;
unsigned int next_key_ = 0;
cudaStream_t* BytePSGlobal::_reduce_stream;
cudaStream_t* BytePSGlobal::_broadcast_stream;
cudaStream_t* BytePSGlobal::_copy_device2host_stream;
cudaStream_t* BytePSGlobal::_copy_host2device_stream;
ncclUniqueId* BytePSGlobal::_nccl_id;
ncclComm_t BytePSGlobal::_nccl_comm;

BytePSScheduledQueue* BytePSGlobal::GetScheduledQueue(QueueType queueType) {
    return (BytePSScheduledQueue*)_queues[queueType];
}

bool BytePSGlobal::IsRootDevice() {
    return _is_root_device;
}

int BytePSGlobal::GetRoot() {
    return _local_size-1;
}

void* BytePSGlobal::CreateScheduledQueue(QueueType queueType) {
    std::lock_guard<std::mutex> lock(_queues_mutex[queueType]);
    if (!_queues[queueType]) {
        _queues[queueType] = new BytePSScheduledQueue(queueType, 34359738368);
    }
}

void BytePSGlobal::Init() {
    std::lock_guard<std::mutex> lock(_init_mutex);
    
    // We only init once
    if (_initialized) {
        return;
    }

#ifdef BYTEPS_USE_MPI
    _comm = std::make_shared<BytePSCommMPI>();
#else
    _comm = std::make_shared<BytePSCommSocket>();
#endif // BYTEPS_USE_MPI

    _comm->init(&_rank, &_size, &_local_rank, &_local_size, &_my_role);

    _is_root_device = (_my_role == LOCAL_ROOT) ? true : false;
    if (getenv("BYTEPS_PARTITION_BYTES")) {
        _partition_bytes = atoi(getenv("BYTEPS_PARTITION_BYTES"));
    }
    BPS_LOG(DEBUG) << "Partition bound set to " << _partition_bytes << " bytes";
    _partition_bytes = _partition_bytes / 8 * 8; // align by 8 (the size of a double or int64)
    BPS_LOG(DEBUG) << "Partition bound is aligned to " << _partition_bytes << " bytes";

    if (_my_role == BytePSRole::LOCAL_ROOT) { // only the root need to do networking
        // init low-level ps implementation
        _ps = new ps::KVWorker<char>(0, 0);
        ps::StartAsync(0, "byteps\0");
        if (!ps::Postoffice::Get()->is_recovery()) {
            ps::Postoffice::Get()->Barrier(
                0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
        }
    }

    CUDA_CALL(cudaSetDevice(_local_rank)); // set to associated GPU
    _reduce_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    _broadcast_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    _copy_host2device_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    _copy_device2host_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    CUDA_CALL(cudaStreamCreateWithFlags(_reduce_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_broadcast_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_copy_host2device_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_copy_device2host_stream, cudaStreamNonBlocking));

    for (int i = 0; i < QueueNum; i++) {
        BPS_LOG(DEBUG) << "Create schedule queue " << i;
        auto type = static_cast<QueueType>(i);
        BytePSGlobal::CreateScheduledQueue(type);
    }

    _initialized = true;
    BPS_LOG(DEBUG) << "Inited rank=" << _rank << " local_rank=" << _local_rank
               << " size=" << _size << " local_size=" << _local_size;

    // init and sycn NCCL id using out-of-band socket
    _nccl_id = (ncclUniqueId*) malloc(sizeof(ncclUniqueId));
    if (_is_root_device) { // only root create nccl id
        NCCLCHECK(ncclGetUniqueId(_nccl_id));
        // the log is just for debug, the actual length of nccl id is 128
        BPS_LOG(DEBUG) << "root NCCL id is " << (*(long long int*)_nccl_id);

        _comm->broadcastSignal(_local_rank, _nccl_id, sizeof(ncclUniqueId), BytePSCommFlag::ROOT_SEND_TO_RECV);
    } else {
        int src;
        int rc = _comm->recvSignal(&src, _nccl_id, sizeof(ncclUniqueId), BytePSCommFlag::NON_ROOT_RECV);
        BPS_CHECK_EQ(rc, sizeof(ncclUniqueId)) << rc << ", " << sizeof(ncclUniqueId);
        BPS_LOG(DEBUG) << "recv root NCCL id " << (*(long long int*)_nccl_id)
                       << ", local_rank=" << _local_rank;
    }

    //initializing NCCL rank
    NCCLCHECK(ncclCommInitRank(&_nccl_comm, _local_size, *_nccl_id, _local_rank));

    return;
}

void BytePSGlobal::Start(LoopFunction* func) {
    // Start background threads
    for (int i = 0; i < ThreadNum; i++) {
        _threads[i] = new std::thread(func[i]);
        BPS_LOG(DEBUG) << "Background thread " << i << " starts.";
    }
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
    for (int i = 0; i < ThreadNum; i++) {
        if (_threads[i]->joinable()) {
            _threads[i]->join();
            delete _threads[i];
        }
    }
    ps::Finalize(0, false);

    cudaStreamDestroy(*_reduce_stream);
    cudaStreamDestroy(*_broadcast_stream);

    for (auto &it:_name_to_cxt) {
        if (it.second.cpubuff && !it.second.reuse_buff) {
            CUDA_CALL(cudaFreeHost(it.second.cpubuff));
        }
    }
    return;
}

BPSContext& BytePSGlobal::GetContextFromName(const std::string &name) {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    BPS_CHECK(_name_to_cxt.find(name) != _name_to_cxt.end()) << name << " is not initialized";
    return _name_to_cxt[name];
}

bool BytePSGlobal::IsTensorInitialized(const std::string &name, size_t size) {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    BPS_CHECK_GT(size, 0) << "init tensor size not larger than 0, should check this";

    if (_name_to_cxt.find(name) == _name_to_cxt.end()) {
        _name_to_cxt[name].initialized = false;

        // _name_to_cxt[name].cpubuff will be inited later
        _name_to_cxt[name].buff_len = size;
        auto accumulated = 0;
        while (accumulated < size) {
            _name_to_cxt[name].key_list.push_back((ps::Key) next_key_++);
            accumulated += ((size - accumulated) > _partition_bytes) ? _partition_bytes : (size - accumulated);
        }

        BPS_LOG(DEBUG) << name << " partitioned to "
                       << _name_to_cxt[name].key_list.size() << " part(s)"
                       << ", total_len=" << size
                       << ", key_range=["
                       << _name_to_cxt[name].key_list.front()
                       << ", "
                       << _name_to_cxt[name].key_list.back()
                       << "]";
        return false;
    }
    return true;
}

PSKV& BytePSGlobal::EncodeDefaultKey(int key, size_t len) {
    _encode_mutex.lock();
    PSKV& pskv = ps_kv_[key];
    _encode_mutex.unlock();
    if (!pskv.keys.empty()) {
        BPS_CHECK_EQ(static_cast<size_t>(pskv.size), len)
            << "The value size cannot be changed " << len
            << ". Key is " << key;
    } else {
        auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
        const int num_servers = krs.size();
        BPS_CHECK_GT(num_servers, 0);
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        BPS_LOG(DEBUG) << "key " << key << " assigned to server " << server;
        ps::Key ps_key = krs[server].begin() + key;
        BPS_CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(len);
        pskv.size = len;
    }
    return pskv;
}

uint32_t BytePSGlobal::GetTensorCount() {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    return BytePSGlobal::_name_to_cxt.size();
}

cudaStream_t* BytePSGlobal::GetReduceStream() {
    return BytePSGlobal::_reduce_stream;
}

cudaStream_t* BytePSGlobal::GetBroadcastStream() {
    return BytePSGlobal::_broadcast_stream;
}

cudaStream_t* BytePSGlobal::GetCopyDevice2HostStream() {
    return BytePSGlobal::_copy_device2host_stream;
}

cudaStream_t* BytePSGlobal::GetCopyHost2DeviceStream() {
    return BytePSGlobal::_copy_host2device_stream;
}

// below are methods for accessing/modifying the _ready_table
bool BytePSGlobal::IsKeyReady(int key) {
    std::lock_guard<std::mutex> lock(_table_mutex);
    return _ready_table[key] == (_local_size - 1);
}

int BytePSGlobal::AddReadyCount(int key) {
    std::lock_guard<std::mutex> lock(_table_mutex);
    BPS_CHECK_LT(_ready_table[key], _local_size-1)
        << _ready_table[key] << ", " << (_local_size-1);
    return ++_ready_table[key];
}

void BytePSGlobal::ClearReadyCount(int key) {
    std::lock_guard<std::mutex> lock(_table_mutex);
    _ready_table[key] = 0;
}

} // namespace common
} // namespace byteps
