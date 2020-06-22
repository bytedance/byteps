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

#include <cstdlib>
#include "common.h"

namespace byteps {
namespace sparse {

int BytePSSparseCommon::_local_size;
int BytePSSparseCommon::_num_worker;
int BytePSSparseCommon::_worker_id;
int BytePSSparseCommon::_global_size;
ps::KVWorker<char>* BytePSSparseCommon::_ps;

void BytePSSparseCommon::Init() {
  CHECK(getenv("BYTEPS_LOCAL_SIZE")) << "error: env BYTEPS_LOCAL_SIZE not set";
  CHECK(getenv("DMLC_WORKER_ID")) << "error: env DMLC_WORKER_ID not set";
  CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";

  _local_size = atoi(getenv("BYTEPS_LOCAL_SIZE"));
  _num_worker = atoi(getenv("DMLC_NUM_WORKER"));
  _worker_id = atoi(getenv("DMLC_WORKER_ID"));
  _global_size = _num_worker * _local_size;

  LOG(INFO) << "local_size=" << _local_size 
      << ", global_size=" << _global_size
      << ", num_worker=" << _num_worker
      << ", worker_id=" << _worker_id;

  // Launch ps-lite if needs distributed training
  if (BytePSSparseCommon::IsDistributed()) {
    // Init worker
    _ps = new ps::KVWorker<char>(0, 0);
    ps::Start(0, "byteps\0");
  }

}


} // namespace sparse
} // namespace byteps 
