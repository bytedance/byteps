// Copyright 2020 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include "ps/ps.h"

namespace byteps {
namespace sparse {

class BytePSSparseCommon {
 public:
  static int GetLocalSize() { return _local_size; }
  static int GetGlobalSize() { return _global_size; }
  static int GetWorkerID() { return _worker_id; }
  static int GetNumWorker() { return _num_worker; }
  static bool IsDistributed() { return _num_worker > 1; }
  static void Init();
  static ps::KVWorker<char>* GetPS() { return _ps; }
  static void AllGather(std::vector<std::vector<int>> src);

 private:
  static int _local_size;
  static int _global_size;
  static int _worker_id;
  static int _num_worker;
  static ps::KVWorker<char>* _ps;

}; // class BytePSSparseCommon


} // namespace sparse
} // namespace byteps 
