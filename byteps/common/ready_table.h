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

#ifndef BYTEPS_READY_TABLE_H
#define BYTEPS_READY_TABLE_H

#include <mutex>
#include <thread>
#include <unordered_map>

namespace byteps {
namespace common {

class ReadyTable {

public:
    ReadyTable(int ready_count) { _ready_count = ready_count; }
    // methods to access or modify the _ready_table
    bool IsKeyReady(int key);
    int AddReadyCount(int key);
    void ClearReadyCount(int key);

private:
    // (key, ready_signal_count) pair, only valid for root device
    std::unordered_map<int, int> _ready_table;
    // use this mutex to access/modify the _ready_table
    std::mutex _table_mutex;
    int _ready_count;

};


} // namespace common
} // namespace byteps

#endif // BYTEPS_READY_TABLE_H