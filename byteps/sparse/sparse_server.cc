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

#include "sparse_server.h"

namespace byteps {
namespace sparse {

using namespace ps;

static ps::KVServer<char>* _byteps_server;
static std::unordered_map<uint64_t, KVPairs<char>> _init_bufferLengths;

template <typename Val>
void BytepsSparseHandler(const ps::KVMeta &req_meta, 
                         const ps::KVPairs<Val> &req_data, 
                         ps::KVServer<Val> *server) {
  uint64_t key = DecodeKey(req_data.keys[0]);
  if (req_meta.push) {
    CHECK(req_data.lens.size());
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]) 
        << "key=" << key << ", " 
        << req_data.vals.size() << ", " 
        << req_data.lens[0];

    auto recved = reinterpret_cast<char*>(req_data.vals.data());

    size_t len = (size_t) req_data.vals.size();
    if (_init_bufferLengths.find(key) == _init_bufferLengths.end()) {
      AllocMemoryAndCreateSarray(_init_bufferLengths[key].keys, (ps::Key*)&req_data.keys[0], 1);
      AllocMemoryAndCreateSarray(_init_bufferLengths[key].vals, recved, len);
      AllocMemoryAndCreateSarray(_init_bufferLengths[key].lens, (int*)&len, 1);
    }

    LOG(INFO) << "key=" << key << "\t" 
        << "len=" << len << "\t"
        << ((int*)recved)[0] << " " << ((int*)recved)[1]
        << "\n";
    
    // send push response (empty payload)
    ps::KVPairs<char> res;
    server->Response(req_meta, res);
  } else { // pull 
    CHECK(_init_bufferLengths.find(key) != _init_bufferLengths.end()) << key;
    server->Response(req_meta, _init_bufferLengths[key]);
  }
}

void InitServer() {
  sharedMemoryInfo info;
  BPS_CHECK_EQ(sharedMemoryOpen(bpsShmName, sizeof(shmStruct), &info), 0);
  auto shm = (volatile shmStruct *)info.addr;

  // init ps-lite instance
  _byteps_server = new ps::KVServer<char>(0);
  _byteps_server->set_request_handle(BytepsSparseHandler<char>);
  ps::StartAsync(0, "byteps_server\0");
  if (!Postoffice::Get()->is_recovery()) {
    ps::Postoffice::Get()->Barrier(0,
      ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }
}

void StopServer() {
  // clean the server resource
  ps::Finalize(0, true);
  if (_byteps_server) {
    delete _byteps_server;
    _byteps_server = nullptr;
  }
}

extern "C" void bytepsSparseServer() {
  InitServer();
  StopServer();
}


} // namespace sparse
} // namespace byteps