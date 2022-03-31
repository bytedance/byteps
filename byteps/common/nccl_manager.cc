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

#include "nccl_manager.h"
#include "global.h"
#include "logging.h"

namespace byteps {
namespace common {

void NcclGroupEntry::RecordEvents() {
  for (size_t i = 0; i < tasks.size(); i++) {
    cudaEvent_t event;
    CUDA_CALL(cudaEventCreateWithFlags(
        &event, cudaEventBlockingSync | cudaEventDisableTiming));
    CUDA_CALL(
        cudaEventRecord(event, BytePSGlobal::GetNccl()->GetStream(
                                   tasks[i]->key, queues[i]->getQueueType())));
    _events.push_back(event);
  }
}

void NcclGroupEntry::SynchronizeEvents() {
  for (size_t i = 0; i < tasks.size(); i++) {
    CUDA_CALL(cudaEventSynchronize(_events[i]));
  }
}

void NcclGroupEntry::DestroyEvents() {
  for (size_t i = 0; i < tasks.size(); i++) {
    CUDA_CALL(cudaEventDestroy(_events[i]));
  }
}

NcclManager::NcclManager(std::shared_ptr<BytePSComm> comm) {
  _global_comm = comm;
  InitGlobalEnv();
  ConstructRings();
  return;
}

ncclComm_t NcclManager::GetComm(uint64_t key, QueueType op) {
  return _nccl_comm[key % _nccl_num_rings];
}

cudaStream_t NcclManager::GetStream(uint64_t key, QueueType op) {
  return _nccl_stream[key % _nccl_num_rings];
}

int NcclManager::GetRoot(uint64_t key, QueueType op) {
  return _nccl_pcie_size - 1;
}

int NcclManager::GetRank(uint64_t key, QueueType op) {
  return BytePSGlobal::GetLocalRank() % _nccl_pcie_size;
}

bool NcclManager::IsSignalRoot() {
  return _signal_comm->getRoot() == BytePSGlobal::GetLocalRank();
}

void NcclManager::ConstructRings() {
  std::string log_string("Constructing NCCL communicators.");
  auto local_rank = BytePSGlobal::GetLocalRank();
  std::vector<int> peers;
  int first_peer = local_rank / _nccl_pcie_size * _nccl_pcie_size;
  for (int i = first_peer; i < first_peer + (int)_nccl_pcie_size; i++) {
    peers.push_back(i);
    log_string = log_string + " " + std::to_string(i);
  }
  _signal_comm = std::make_shared<BytePSCommSocket>(_global_comm,
                                                    std::string("nccl"), peers);
  BPS_LOG(DEBUG) << log_string;

  // init and sycn NCCL-reduce-id using out-of-band socket
  _nccl_id = (ncclUniqueId*)malloc(sizeof(ncclUniqueId) * _nccl_num_rings);
  _nccl_comm = (ncclComm_t*)malloc(sizeof(ncclComm_t) * _nccl_num_rings);
  _nccl_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * _nccl_num_rings);
  _nccl_size = _nccl_pcie_size;
  int greatest_priority;
  CUDA_CALL(cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));

  for (size_t i = 0; i < _nccl_num_rings; i++) {
    auto nccl_id = _nccl_id + i;
    auto nccl_comm = _nccl_comm + i;
    auto nccl_stream = _nccl_stream + i;

    // synchronize NCCL IDs
    if (local_rank == _signal_comm->getRoot()) {  // only root create nccl id
      NCCLCHECK(ncclGetUniqueId(nccl_id));
      // the log is just for debug, the actual length of nccl id is 128
      BPS_LOG(DEBUG) << "root nccl_id is " << (*(long long int*)nccl_id);
      // TODO: change to BytePSCommSignal format
      _signal_comm->broadcastSignal(nccl_id, sizeof(ncclUniqueId));

    } else {
      int src;
      // TODO: change to recvSignalFromRoot after using BytePSCommSignal format
      int rc = _signal_comm->recvSignal(&src, nccl_id, sizeof(ncclUniqueId));
      BPS_CHECK_EQ(rc, sizeof(ncclUniqueId))
          << rc << ", " << sizeof(ncclUniqueId);
      BPS_LOG(DEBUG) << "recv nccl_id is " << (*(long long int*)nccl_id)
                     << ", local_rank=" << local_rank;
    }

    // initialize NCCL rank
    auto rank = local_rank % _nccl_pcie_size;
    NCCLCHECK(ncclCommInitRank(nccl_comm, _nccl_pcie_size, *nccl_id, rank));

    // initialize CUDA streams for NCCL
    CUDA_CALL(cudaStreamCreateWithPriority(nccl_stream, cudaStreamNonBlocking,
                                           greatest_priority));
    CUDA_CALL(cudaStreamSynchronize(*nccl_stream));
  }
}

void NcclManager::InitGlobalEnv() {  // init all global env/param here
  _nccl_group_size =
      (getenv("BYTEPS_NCCL_GROUP_SIZE") ? atoi(getenv("BYTEPS_NCCL_GROUP_SIZE"))
                                        : 4);
  BPS_LOG(DEBUG) << "nccl_group_size"
                 << " set to " << _nccl_group_size;

  _nccl_pcie_size = (getenv("BYTEPS_PCIE_SWITCH_SIZE")
                         ? atoi(getenv("BYTEPS_PCIE_SWITCH_SIZE"))
                         : 8);
  auto local_size = BytePSGlobal::GetLocalSize();
  _nccl_pcie_num = local_size / _nccl_pcie_size;
  if (!_nccl_pcie_num) {
    _nccl_pcie_size = local_size;
    _nccl_pcie_num = 1;
  } else {
    if (local_size % _nccl_pcie_size) {
      BPS_LOG(WARNING) << "BytePS does not support unbalanced PCIe switches.";
      _nccl_pcie_size = local_size;
      _nccl_pcie_num = 1;
    }
  }

  BPS_LOG(DEBUG) << "nccl_pcie_size"
                 << " set to " << _nccl_pcie_size;
  BPS_LOG(DEBUG) << "nccl_pcie_num"
                 << " set to " << _nccl_pcie_num;

  _nccl_num_rings =
      (getenv("BYTEPS_NCCL_NUM_RINGS") ? atoi(getenv("BYTEPS_NCCL_NUM_RINGS"))
                                       : 1);
  if (_nccl_num_rings != 1) {
    BPS_LOG(INFO) << "nccl_num_rings is not 1, it can improve NCCL performance, "
                  << "but may leads to occasional hanging problem";
  }
  return;
}

void NcclManager::EnqueueGroup(std::shared_ptr<NcclGroupEntry> e) {
  std::lock_guard<std::mutex> lock(_nccl_mutex);
  _nccl_pipeline.push(e);
  return;
}

std::shared_ptr<NcclGroupEntry> NcclManager::DequeueGroup() {
  std::lock_guard<std::mutex> lock(_nccl_mutex);
  if (!_nccl_pipeline.size()) {
    return nullptr;
  }
  auto r = _nccl_pipeline.front();
  _nccl_pipeline.pop();
  return r;
}

// Example:
// 4 reduce rings:
// 0 1 2 3 | 4 5 6 7
// 1 2 3 0 | 5 6 7 4
// 2 3 0 1 | 6 7 4 5
// 3 0 1 2 | 7 4 5 6
//
// reduce
// 1st ring, 0->1->2->3->cpubuff->4->5->6->7->cpubuff,
// 4->5->6->7->cpubuff->0->1->2->3->cpubuff 2nd ring,
// 1->2->3->0->cpubuff->5->6->7->4->cpubuff,
// 5->6->7->4->cpubuff->1->2->3->0->cpubuff 3rd ring,
// 2->3->0->1->cpubuff->6->7->4->5->cpubuff,
// 6->7->4->5->cpubuff->2->3->0->1->cpubuff 4th ring,
// 3->0->1->2->cpubuff->7->4->5->6->cpubuff,
// 7->4->5->6->cpubuff->3->0->1->2->cpubuff
//
// 4 broadcast rings (reverse of reduce rings)
// 7 6 5 4 | 3 2 1 0
// 4 7 6 5 | 0 3 2 1
// 5 4 7 6 | 1 0 3 2
// 6 5 4 7 | 2 1 0 3
//
// broadcast
// 1st ring, cpubuff->7->6->5->4->cpubuff->3->2->1->0,
// cpubuff->3->2->1->0->cpubuff->7->6->5->4 2nd ring,
// cpubuff->4->7->6->5->cpubuff->0->3->2->1,
// cpubuff->0->3->2->1->cpubuff->4->7->6->5 3rd ring,
// cpubuff->5->4->7->6->cpubuff->1->0->3->2,
// cpubuff->1->0->3->2->cpubuff->5->4->7->6 4th ring,
// cpubuff->6->5->4->7->cpubuff->2->1->0->3,
// cpubuff->2->1->0->3->cpubuff->6->5->4->7
//
void NcclManagerExpr::ConstructRings() {
  _signal_comm = _global_comm;
  BPS_LOG(DEBUG) << "Constructing NCCL Reduce communicators.";
  for (size_t i = 0; i < _nccl_pcie_size; i++) {
    _rings.push_back(std::vector<int>());
    std::string log("");
    for (size_t j = 0; j < _nccl_pcie_num; j++) {
      for (size_t k = 0; k < _nccl_pcie_size; k++) {
        int rank = (k + i) % _nccl_pcie_size + j * _nccl_pcie_size;
        _rings[i].push_back(rank);
        log = log + std::to_string(rank) + ' ';
      }
    }
    BPS_LOG(DEBUG) << log;
  }
  BPS_LOG(DEBUG) << "Constructing NCCL Broadcast communicators.";
  for (size_t i = 0; i < _nccl_pcie_size; i++) {
    _rings.push_back(std::vector<int>());
    std::string log("");
    for (int j = 0; j < BytePSGlobal::GetLocalSize(); j++) {
      int rank = _rings[i][BytePSGlobal::GetLocalSize() - j - 1];
      _rings[i + _nccl_pcie_size].push_back(rank);
      log = log + std::to_string(rank) + ' ';
    }
    BPS_LOG(DEBUG) << log;
  }
  auto local_size = BytePSGlobal::GetLocalSize();
  auto local_rank = BytePSGlobal::GetLocalRank();
  // init and sycn NCCL-reduce-id using out-of-band socket
  _nccl_id = (ncclUniqueId*)malloc(sizeof(ncclUniqueId) * _nccl_pcie_size * 2);
  _nccl_comm = (ncclComm_t*)malloc(sizeof(ncclComm_t) * _nccl_pcie_size * 2);
  _nccl_stream =
      (cudaStream_t*)malloc(sizeof(cudaStream_t) * _nccl_pcie_size * 2);
  int greatest_priority;
  CUDA_CALL(cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));

  for (size_t i = 0; i < _nccl_pcie_size * 2; i++) {
    auto nccl_id = _nccl_id + i;
    auto nccl_comm = _nccl_comm + i;
    auto nccl_stream = _nccl_stream + i;

    // synchronize NCCL IDs
    if (BytePSGlobal::IsRootDevice()) {  // only root create nccl id
      NCCLCHECK(ncclGetUniqueId(nccl_id));
      // the log is just for debug, the actual length of nccl id is 128
      BPS_LOG(DEBUG) << "root nccl_id is " << (*(long long int*)nccl_id);
      _signal_comm->broadcastSignal(nccl_id, sizeof(ncclUniqueId));
    } else {
      int src;
      int rc = _signal_comm->recvSignal(&src, nccl_id, sizeof(ncclUniqueId));
      BPS_CHECK_EQ(rc, sizeof(ncclUniqueId))
          << rc << ", " << sizeof(ncclUniqueId);
      BPS_LOG(DEBUG) << "recv nccl_id is " << (*(long long int*)nccl_id)
                     << ", local_rank=" << local_rank;
    }

    // initialize NCCL rank
    auto it = std::find(_rings[i].begin(), _rings[i].end(), local_rank);
    auto rank = std::distance(_rings[i].begin(), it);
    NCCLCHECK(ncclCommInitRank(nccl_comm, local_size, *nccl_id, rank));

    // initialize CUDA streams for NCCL
    CUDA_CALL(cudaStreamCreateWithPriority(nccl_stream, cudaStreamNonBlocking,
                                           greatest_priority));
    CUDA_CALL(cudaStreamSynchronize(*nccl_stream));
  }
  return;
}

ncclComm_t NcclManagerExpr::GetComm(uint64_t key, QueueType op) {
  auto offset = (op == REDUCE) ? 0 : _nccl_pcie_size;
  return _nccl_comm[key % _nccl_pcie_size + offset];
}

cudaStream_t NcclManagerExpr::GetStream(uint64_t key, QueueType op) {
  auto offset = (op == REDUCE) ? 0 : _nccl_pcie_size;
  return _nccl_stream[key % _nccl_pcie_size + offset];
}

int NcclManagerExpr::GetRoot(uint64_t key, QueueType op) {
  int comm_index = key % _nccl_pcie_size;
  int pcie_index = key % (_nccl_pcie_size * _nccl_pcie_num) / _nccl_pcie_size;
  int root = -1;
  if (op == REDUCE) {
    int root_index = (_nccl_pcie_num - pcie_index) * _nccl_pcie_size - 1;
    root = _rings[comm_index][root_index];
  } else {
    BPS_CHECK_EQ(op, BROADCAST) << "Unknown OP for NcclManager.";
    int root_index = pcie_index * _nccl_pcie_size;
    root = _rings[comm_index + _nccl_pcie_size][root_index];
  }
  BPS_CHECK_GT(root, -1);
  return root;
}

int NcclManagerExpr::GetRank(uint64_t key, QueueType op) {
  auto offset = (op == REDUCE) ? 0 : _nccl_pcie_size;
  auto i = key % _nccl_pcie_size + offset;
  auto it = std::find(_rings[i].begin(), _rings[i].end(),
                      BytePSGlobal::GetLocalRank());
  auto rank = std::distance(_rings[i].begin(), it);
  return rank;
}

}  // namespace common
}  // namespace byteps
