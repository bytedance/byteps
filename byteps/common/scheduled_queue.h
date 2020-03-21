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

namespace byteps {
namespace common {

class BytePSScheduledQueue {
 public:
  BytePSScheduledQueue(QueueType type);
  QueueType getQueueType() { return _qt; }
  void addTask(std::shared_ptr<TensorTableEntry>);
  void recorderTs(std::shared_ptr<TensorTableEntry>);
  std::shared_ptr<TensorTableEntry> getTask();
  std::shared_ptr<TensorTableEntry> getTask(uint64_t key);
  uint32_t pendingSize();
  void reportFinish(int size);
  void reset(uint64_t key, int cnt);

 private:
  // TODO: use priority queue or heap
  std::vector<std::shared_ptr<TensorTableEntry>> _sq;
  std::mutex _mutex;
  uint64_t _credits;
  bool _is_scheduled;
  QueueType _qt;
  ReadyTable *_rt;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SCHEDULED_QUEUE_H
