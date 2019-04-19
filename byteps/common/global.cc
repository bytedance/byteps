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

namespace byteps {
namespace common {

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
    std::lock_guard<std::mutex> lock(_mutex);
    _sq.push_back(entry);
    return;
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
    std::lock_guard<std::mutex> lock(_mutex);
    auto front = _sq.front();
    _sq.pop_front();
    return front;
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::peekTask() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _sq.front();
}

uint32_t BytePSScheduledQueue::pendingSize() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _sq.size();
}

void BytePSScheduledQueue::reportFinish(std::shared_ptr<TensorTableEntry> e) {
    // TODO: return credit based on TensorTableEntry
    _finished++;
    return;
}

// Define and init global variables
std::mutex BytePSGlobal::_init_mutex;
volatile bool BytePSGlobal::_initialized = false;
volatile bool BytePSGlobal::_should_shutdown = false;

int BytePSGlobal::_rank = 0;
int BytePSGlobal::_local_rank = 0;
int BytePSGlobal::_size = 1;
int BytePSGlobal::_local_size = 1;
MPI_Comm BytePSGlobal::_local_comm;
MPI_Comm BytePSGlobal::_global_comm;

volatile BytePSScheduledQueue* BytePSGlobal::_queues[QueueNum] = {NULL};
std::mutex BytePSGlobal::_queues_mutex[QueueNum];
std::thread* BytePSGlobal::_threads[QueueNum] = {NULL};

ps::KVWorker<char>* BytePSGlobal::_ps = NULL;
std::mutex BytePSGlobal::_encode_mutex;
std::unordered_map<std::string, BPSContext> BytePSGlobal::_name_to_cxt;
unsigned int next_key_ = 0;

cudaStream_t* BytePSGlobal::_reduce_stream;
cudaStream_t* BytePSGlobal::_broadcast_stream;

BytePSScheduledQueue* BytePSGlobal::GetScheduledQueue(QueueType queueType) {
    return (BytePSScheduledQueue*)_queues[queueType];
}

void* BytePSGlobal::CreateScheduledQueue(QueueType queueType) {
    std::lock_guard<std::mutex> lock(_queues_mutex[queueType]);
    if (!_queues[queueType]) {
        _queues[queueType] = new BytePSScheduledQueue();
    }
}

void BytePSGlobal::_InitComm() {

    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm_dup(MPI_COMM_WORLD, &_global_comm);

    // Get MPI size to determine how many tensors to wait for before reducing.
    MPI_Comm_rank(_global_comm, &_rank);
    MPI_Comm_size(_global_comm, &_size);

    // Determine local rank by querying the local communicator.
    MPI_Comm_split_type(_global_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &_local_comm);
    MPI_Comm_rank(_local_comm, &_local_rank);
    MPI_Comm_size(_local_comm, &_local_size);
}

void BytePSGlobal::Init() {
    std::lock_guard<std::mutex> lock(_init_mutex);
    
    // We only init once
    if (_initialized) {
        return;
    }

    _InitComm();

    // init low-level ps implementation
    _ps = new ps::KVWorker<char>(0, 0);
    ps::StartAsync(0, "byteps\0");
    if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(
            0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }

    _reduce_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    _broadcast_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    cudaStreamCreateWithFlags(_reduce_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(_broadcast_stream, cudaStreamNonBlocking);

    for (int i = 0; i < QueueNum; i++) {
        BPS_LOG(DEBUG) << "Create schedule queue " << i;
        auto type = static_cast<QueueType>(i);
        BytePSGlobal::CreateScheduledQueue(type);
    }

    _initialized = true;
    BPS_LOG(DEBUG) << "Inited rank=" << _rank << " local_rank=" << _local_rank
               << " size=" << _size << " local_size=" << _local_size;
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
    ps::Finalize(0, true);

    cudaStreamDestroy(*_reduce_stream);
    cudaStreamDestroy(*_broadcast_stream);

    return;
}

BPSContext& BytePSGlobal::GetContextFromName(const std::string &name) {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    return _name_to_cxt[name];
}

bool BytePSGlobal::IsTensorInitialized(const std::string &name) {
    std::lock_guard<std::mutex> lock(_encode_mutex);
    if (_name_to_cxt.find(name) == _name_to_cxt.end()) {
        _name_to_cxt[name].key = (ps::Key) next_key_++;
        return false;
    }
    return true;
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

} // namespace common
} // namespace byteps
