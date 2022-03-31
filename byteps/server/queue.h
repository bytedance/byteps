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

#ifndef BYTEPS_SERVER_QUEUE_H
#define BYTEPS_SERVER_QUEUE_H

#include <vector>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <algorithm>

namespace byteps {
namespace server {

/**
 * \brief thread-safe queue allowing push and waited pop
 */
class PriorityQueue {
 public:
  PriorityQueue(bool is_schedule) {
    enable_schedule_ = is_schedule;
    if (enable_schedule_) {
      std::make_heap(queue_.begin(), queue_.end(),
        [this](const BytePSEngineMessage& a, const BytePSEngineMessage& b) {
          return ComparePriority(a, b);
        }
      );
    }
  }
  ~PriorityQueue() { }

  /**
   * \brief push an value and sort using heap. threadsafe.
   * \param new_value the value
   */
  void Push(BytePSEngineMessage new_value) {
    mu_.lock();
    queue_.push_back(std::move(new_value));
    if (enable_schedule_) {
      ++push_cnt_[new_value.key];
      std::push_heap(queue_.begin(), queue_.end(),
        [this](const BytePSEngineMessage& a, const BytePSEngineMessage& b) {
          return ComparePriority(a, b);
        }
      );
    }
    mu_.unlock();
    cond_.notify_all();
  }

  /**
   * \brief wait until pop an element from the beginning, threadsafe
   * \param value the poped value
   */
  void WaitAndPop(BytePSEngineMessage* value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this]{return !queue_.empty();});
    if (enable_schedule_) {
      std::pop_heap(queue_.begin(), queue_.end(),
        [this](const BytePSEngineMessage& a, const BytePSEngineMessage& b) {
          return ComparePriority(a, b);
        }
      );
      *value = queue_.back();
      queue_.pop_back();
    } else {
      *value = std::move(queue_.front());
      queue_.erase(queue_.begin());
    }
  }

  void ClearCounter(uint64_t key) {
    if (!enable_schedule_) return;
    std::unique_lock<std::mutex> lk(mu_);
    push_cnt_[key] = 0;
  }

  bool ComparePriority(const BytePSEngineMessage& a, const BytePSEngineMessage& b) {
    if (push_cnt_[a.key] == push_cnt_[b.key]) {
      return (a.id > b.id);
    } else {
      return (push_cnt_[a.key] > push_cnt_[b.key]);
    }
  }

 private:
  mutable std::mutex mu_;
  std::vector<BytePSEngineMessage> queue_;
  std::condition_variable cond_;
  std::unordered_map<uint64_t, uint64_t> push_cnt_;
  volatile bool enable_schedule_ = false;
};

}  // namespace server
}  // namespace byteps

#endif  // BYTEPS_SERVER_QUEUE_H