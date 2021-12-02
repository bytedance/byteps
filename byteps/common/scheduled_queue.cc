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

#include "scheduled_queue.h"

#include <algorithm>

#include "global.h"
#include "logging.h"

namespace byteps {
namespace common {

BytePSScheduledQueue::BytePSScheduledQueue(QueueType type, bool lockless) : _spsc(32768) {
  _lockless = lockless;

#if BYTEPS_BUILDING_CUDA == 1
  size_t credit_in_partition = BytePSGlobal::GetNccl()->GetGroupSize() + 1;
  if (type == REDUCE && BytePSGlobal::GetNccl()->IsSignalRoot()) {
    _is_scheduled = true;
  } else {
    _is_scheduled = false;
  }
#else
  // FIXME: what should be the credit if CUDA is not available?
  size_t credit_in_partition = 0;
  _is_scheduled = false;
#endif

  auto byteps_scheduling_credit = getenv("BYTEPS_SCHEDULING_CREDIT");
  credit_in_partition =
      byteps_scheduling_credit ? atoi(byteps_scheduling_credit) : 0;
  if (!credit_in_partition) {  // disable scheduling by default
    _is_scheduled = false;
  }

  _qt = type;
  _credits = _is_scheduled
                 ? BytePSGlobal::GetPartitionBound() * credit_in_partition
                 : 34359738368;  // 32GB, basically disabling credit control

  BPS_CHECK(!(_lockless && _is_scheduled))
        << "A queue cannot be lockless and also scheduled. Queue: " << LogStrings[_qt];

  _rt = nullptr;

  switch (_qt) {
    case P2P_GROUP_COPYH2D: {
      _rt = BytePSGlobal::GetP2PGroupCopyTable();
      break;
    }
    case RECV: {
      _rt = BytePSGlobal::GetP2PCopyTable();
      break;
    }    
    case P2P_PULL_RESPONSE: {
      _rt = BytePSGlobal::GetP2PPullResponseTable();
      break;
    }
    case P2P_WAIT_ACK: {
      _rt = BytePSGlobal::GetP2PAckTable();
      break;
    }
    case REDUCE:
#if BYTEPS_BUILDING_CUDA == 1
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetReduceTable();
      }
#else
      _rt = nullptr;
      BPS_LOG(WARNING) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
      break;
    case PCIE_REDUCE:
#if BYTEPS_BUILDING_CUDA == 1
      if (BytePSGlobal::IsCrossPcieSwitch()) {
        if (BytePSGlobal::GetCpuReducer()->isRoot()) {
          _rt = BytePSGlobal::GetPcieReduceTable();
        }
      }
#else
      _rt = nullptr;
      BPS_LOG(WARNING) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
      break;
    case COMPRESS:
    case PUSH:
      _rt = BytePSGlobal::GetPushTable();
      break;
    case COPYH2D:
      if (!BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetCopyTable();
      }
      break;
    case BROADCAST:
#if BYTEPS_BUILDING_CUDA == 1
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetBroadcastTable();
      }
#else
      _rt = nullptr;
      BPS_LOG(WARNING) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
      break;
    case CPU_REDUCE:
      _rt = BytePSGlobal::GetCpuReduceTable();
      break;
    case CPU_BCAST:
      _rt = BytePSGlobal::GetCpuBcastTable();
      break;
    case CPU_BCAST_FINISH:
      _rt = BytePSGlobal::GetCpuBcastFinishTable();
      break;
    case ALLGATHER:
#if BYTEPS_BUILDING_CUDA == 1
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetAllgatherTable();
      }
#else
      _rt = nullptr;
      BPS_LOG(WARNING) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
      break;
    case ALLGATHER_BCAST:
#if BYTEPS_BUILDING_CUDA == 1
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetAllgatherBcastTable();
      }
#else
      _rt = nullptr;
      BPS_LOG(WARNING) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
      break;
    case ALLGATHER_PULL_RESPONSE:
      _rt = BytePSGlobal::GetAllgatherPullResponseTable();
      break;
    case ALLGATHER_P2P_WAIT_ACK:
      _rt = BytePSGlobal::GetAllgatherAckTable();
      break;
    case ALLGATHER_COPYH2D:
      if (!BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetAllgatherCopyH2DTable();
      }
      break;
    case GDR_WAIT_PUSH_PULL:
      _rt = BytePSGlobal::GetGDRPushPullTable();
      break;
    case GDR_WAIT_ACK:
      _rt = BytePSGlobal::GetGDRAckTable();
      break;
    default:
      break;
  }
  BPS_CHECK(!(_lockless && _rt))
      << "Lockless queue with ready table is not supported yet. Queue: " << LogStrings[_qt];
}

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
  doAddTask(entry, _sq_shared);
}

void BytePSScheduledQueue::addTask(TensorTableEntry* entry) {
  if (_lockless) {
    addTaskLiteLockless(entry);
  } else {
    doAddTask(entry, _sq_lite);
  }
}

template <typename T>
void BytePSScheduledQueue::doAddTask(T& entry, std::vector<T>& sq) {
  std::lock_guard<std::mutex> lock(_mutex);
  sq.push_back(entry);
  if (_is_scheduled) {
    // TODO: below can be optimized to O(n) using insertion sort
    std::sort(
        sq.begin(), sq.end(),
        [](T a, T b) {
          if (a->priority == b->priority) {
            return (a->key < b->key);  // from the first partition to the last
          }
          return (a->priority > b->priority);  // from higher priority to lower
        });
  }
  BPS_CHECK(entry->tensor_name != "");
  BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                 << " addTask: " << entry->tensor_name << " key: " << entry->key
                 << " rank: " << BytePSGlobal::GetLocalRank()
		             << " worker_id: " << BytePSGlobal::GetWorkerID();
  return;
}

// Record the start time of the sub-tasks for all QueueTypes of each partition.
template <typename T>
void BytePSScheduledQueue::doRecordTs(T& task) {
  auto context = task->context;
  // add for profiling
  if (context && context->profile_flag) {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

    auto &queue_list = task->queue_list;
    BPS_CHECK_GE(queue_list.size(), 1);
    auto this_op = queue_list[0];

    BPSCommTime *ret = new BPSCommTime;
    ret->start_t = (long long)(us.count());
    ret->key = task->key;
    ret->type = this_op;
    context->part_comm_time[task->key][this_op].push(ret);
  }
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
  return doGetTask(_sq_shared);
}

TensorTableEntry* BytePSScheduledQueue::getTaskLite() {
  if (_lockless) {
    return getTaskLiteLockless();
  }
  return doGetTask(_sq_lite);
}

template <typename T>
T BytePSScheduledQueue::doGetTask(std::vector<T>& sq) {
  std::lock_guard<std::mutex> lock(_mutex);
  T task;
  // TODO: below can be optimized -- if we take task from the tail, erase() can
  // be faster
  for (auto it = sq.begin(); it != sq.end(); ++it) {
    if ((*it)->ready_event) {
      if (!(*it)->ready_event->Ready()) {
        continue;
      }
    }
    if (_is_scheduled) {
      if ((*it)->len > _credits) {
        continue;
      }
    }
    if (_rt) {
      if (!_rt->IsKeyReady((*it)->key)) {
        continue;
      }
      _rt->ClearReadyCount((*it)->key);
    }
    task = *it;
    sq.erase(it);
    if (_is_scheduled) {
      _credits -= task->len;
    }

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                   << " getTask: " << task->tensor_name << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication traces
    doRecordTs(task);
    return task;
  }
  return nullptr;
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask(uint64_t key) {
  return doGetTask(key, _sq_shared);
}

template <typename T>
T BytePSScheduledQueue::doGetTask(uint64_t key, std::vector<T>& sq) {
  BPS_CHECK(!_is_scheduled);
  BPS_CHECK(!_lockless);
  std::lock_guard<std::mutex> lock(_mutex);
  T task;
  for (auto it = sq.begin(); it != sq.end(); ++it) {
    if ((*it)->ready_event) {
      BPS_CHECK((*it)->ready_event->Ready());
    }
    if ((*it)->key != (uint64_t)key) {
      continue;
    }
    task = *it;
    sq.erase(it);

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                   << " getTask(key): " << task->tensor_name
                   << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication traces
    doRecordTs(task);
    return task;
  }
  return nullptr;
}

void BytePSScheduledQueue::reportFinish(int size) {
  if (_is_scheduled) {
    std::lock_guard<std::mutex> lock(_mutex);
    _credits += size;
  }
  return;
}

void BytePSScheduledQueue::reset(uint64_t key, int cnt) {
  std::lock_guard<std::mutex> lock(_mutex);
  if(_rt) {
    _rt->SetReadyCount(key, cnt);
  }
}

void BytePSScheduledQueue::addTaskLiteLockless(TensorTableEntry* entry) {
  std::lock_guard<std::mutex> lock(_write_mu);
  _spsc.push(entry);
}

TensorTableEntry* BytePSScheduledQueue::getTaskLiteLockless() {
  std::lock_guard<std::mutex> lock(_read_mu);
  if (_spsc.front()) {
    TensorTableEntry* task = *_spsc.front();
    _spsc.pop();
    return task;
  }
  return nullptr;
}

void BytePSScheduledQueue::getPendingTasks(std::unordered_map<uint64_t, TaskMetaMap>* results) {
  std::lock_guard<std::mutex> lock(_mutex);
  BPS_CHECK(!_lockless);
  for (auto& task : _sq_shared) {
    TaskMeta::addPendingTask(task.get(), results);
  }
  for (auto& task : _sq_lite) {
    TaskMeta::addPendingTask(task, results);
  }
}

}  // namespace common
}  // namespace byteps
