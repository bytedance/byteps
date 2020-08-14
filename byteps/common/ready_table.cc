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

#include "ready_table.h"

#include "logging.h"

namespace byteps {
namespace common {

// below are methods for accessing/modifying the _ready_table
bool ReadyTable::IsKeyReady(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  return _ready_table[key] == (_ready_count);
}

int ReadyTable::AddReadyCount(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  BPS_CHECK_LT(_ready_table[key], _ready_count)
      << _table_name << ": " << _ready_table[key] << ", " << (_ready_count);
  return ++_ready_table[key];
}

int ReadyTable::SetReadyCount(uint64_t key, int cnt) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  _ready_table[key] = cnt;
}

void ReadyTable::ClearReadyCount(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  _ready_table[key] = 0;
}

}  // namespace common
}  // namespace byteps
