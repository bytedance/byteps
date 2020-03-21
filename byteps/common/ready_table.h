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

#ifndef BYTEPS_READY_TABLE_H
#define BYTEPS_READY_TABLE_H

#include <mutex>
#include <thread>
#include <unordered_map>

namespace byteps {
namespace common {

class ReadyTable {
 public:
  ReadyTable(int ready_count, const char* name) {
    _ready_count = ready_count;
    _table_name = std::string(name);
  }
  // methods to access or modify the _ready_table
  bool IsKeyReady(uint64_t key);
  int AddReadyCount(uint64_t key);
  int SetReadyCount(uint64_t key, int cnt);
  void ClearReadyCount(uint64_t key);

 private:
  // (key, ready_signal_count) pair, only valid for root device
  std::unordered_map<uint64_t, int> _ready_table;
  // use this mutex to access/modify the _ready_table
  std::mutex _table_mutex;
  int _ready_count;
  std::string _table_name;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_READY_TABLE_H
