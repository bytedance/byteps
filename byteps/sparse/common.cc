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

  // LOG(INFO) << "local_size=" << _local_size 
  //     << ", global_size=" << _global_size
  //     << ", num_worker=" << _num_worker
  //     << ", worker_id=" << _worker_id;

  
  // Launch ps-lite if needs distributed training
  if (BytePSSparseCommon::IsDistributed()) {
    // Init worker
    _ps = new ps::KVWorker<char>(0, 0);
    ps::StartAsync(0, "byteps\0");
    ps::Postoffice::Get()->Barrier(0, 
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }
}

void BytePSSparseCommon::CoordinateDistBufferLens(std::vector<std::vector<int>> src) {
  auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
  const int numServers = krs.size();
  int rank = BytePSSparseCommon::GetWorkerID();
  CHECK_EQ(numServers, (int)src.size()) << numServers << " " << src.size();

  // Prepare vals
  std::vector<ps::SArray<char>> psVals(numServers);
  for (int i = 0; i < numServers; i++) {
    ps::SArray<char> tmp((char*)src[i].data(), numServers * sizeof(int), false);
    psVals[i] = tmp;
  }
  
  // Prepare keys and lens
  std::vector<ps::SArray<ps::Key>> psKeys; 
  std::vector<ps::SArray<int>> psLens; 
  for (int i = 0; i < numServers; i++) {
    int server = i;
    std::vector<ps::Key> tmp1(1, (ps::Key)krs[server].begin() + server);
    ps::SArray<ps::Key> keys(tmp1);
    psKeys.push_back(keys);

    std::vector<int> tmp2(1, (int)src.size() * sizeof(int));
    ps::SArray<int> lens(tmp2);
    psLens.push_back(lens);
  }

  // Push myself
  { 
    int server = rank;
    auto keys = psKeys[server];
    auto vals = psVals[server];
    auto lens = psLens[server];
    _ps->Wait(_ps->ZPush(keys, vals, lens));
  }

  // Pull from others
  for (int i = 0; i < numServers; i++) {
    if (i == rank) continue; // skip myself
    int server = i;
    auto keys = psKeys[server];
    auto vals = psVals[server];
    auto lens = psLens[server];
    _ps->Wait(_ps->ZPull(keys, &vals, &lens));
  }

}


} // namespace sparse
} // namespace byteps 
