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

namespace byteps {
namespace server {

/**
 * \brief thread-safe queue allowing push and waited pop
 */
class PriorityQueue {
 public:
  PriorityQueue(bool is_schedule) { 
    enable_schedule_ = is_schedule;
  }
  ~PriorityQueue() { }

  /**
   * \brief push an value and sort. threadsafe.
   * \param new_value the value
   */
  void Push(BytePSEngineMessage new_value) {
    mu_.lock();
    queue_.push_back(std::move(new_value));
    if (enable_schedule_) {
      push_cnt_[new_value.key] = push_cnt_[new_value.key] % (size_t) ps::NumWorkers() + 1; 
      std::sort(queue_.begin(), queue_.end(),
        [this](const BytePSEngineMessage& a, const BytePSEngineMessage& b) {
          if (push_cnt_[a.key] == push_cnt_[b.key]) {
            // smaller key is dequeued first
            return (a.key < b.key);
          }
          // Dequeue those with more recevied pushes
          return (push_cnt_[a.key] > push_cnt_[b.key]);
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
    *value = std::move(queue_.back());
    queue_.pop_back();
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