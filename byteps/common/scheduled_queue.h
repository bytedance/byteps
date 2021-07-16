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

#ifndef BYTEPS_SCHEDULED_QUEUE_H
#define BYTEPS_SCHEDULED_QUEUE_H

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include "common.h"
#include "ready_table.h"
#include "spsc_queue.h"

namespace byteps {
namespace common {

// T = {std::shared_ptr<TensorTableEntry>, TensorTableEntry*}
// typically, with a TensorTableEntry*, it is created in operations.cc
// during `EnqueueTensor` and destroyed in `FinishOrProceed` in core_loops.cc
class BytePSScheduledQueue {
 public:
  BytePSScheduledQueue(QueueType type, bool lockless = false);
  QueueType getQueueType() { return _qt; }

  // Currently ScheduledQueue provides two sets of interfaces:
  // IF1. addTask and getTask with shared_ptr<TensorTableEntry>
  // IF2. addTaskLite and getTaskLite with TensorTableEntry
  //
  // Note that IF1 uses inputs with shared_ptr, which means the
  // task's life cycle is managed by std::shared_ptr. As of now,
  // push/pull/send/recv tasks uses IF1 with shared_ptr.
  //
  // On the other hand, some operators uses IF2 to enqueue tensor.
  // Not using shared_ptr means that bytePS manages the life cycle
  // of the task. The task usually is allocated in `EnqueueTensorXX`
  // in operations.cc, and freed in `FinishOrProceedLite` in
  // core_loops.cc
  void addTask(std::shared_ptr<TensorTableEntry>);
  void addTask(TensorTableEntry*);
  std::shared_ptr<TensorTableEntry> getTask();
  TensorTableEntry* getTaskLite();
  std::shared_ptr<TensorTableEntry> getTask(uint64_t key);
  void reportFinish(int size);
  void reset(uint64_t key, int cnt);

 private:
  template <typename T>
  void doRecordTs(T&);

  template <typename T>
  void doAddTask(T&, std::vector<T>&);

  template <typename T>
  T doGetTask(std::vector<T>&);

  template <typename T>
  T doGetTask(uint64_t key, std::vector<T>&);

  // TODO: use priority queue or heap
  std::vector<std::shared_ptr<TensorTableEntry>> _sq_shared;
  std::vector<TensorTableEntry*> _sq_lite;

  std::mutex _mutex;
  uint64_t _credits;
  bool _is_scheduled;
  QueueType _qt;
  ReadyTable *_rt;

  // lockless implementation for TensorTableEntry*
  bool _lockless;
  std::mutex _read_mu;
  std::mutex _write_mu;
  rigtorp::SPSCQueue<TensorTableEntry*> _spsc;
  // the lockless implementation only supports TensorTableEntry*
  void addTaskLiteLockless(TensorTableEntry*);
  TensorTableEntry* getTaskLiteLockless();

};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SCHEDULED_QUEUE_H
